"""Kernel feature extraction using Clang/LLVM.

This module provides a command line interface for extracting a series of
structural, control-flow, memory and arithmetic features from a C/C++ source
file. The analysis uses libclang through the clang.cindex bindings. The
resulting feature values are written to a CSV file that shares the basename of
input source.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from clang import cindex


CONTROL_STATEMENT_KINDS = {
    cindex.CursorKind.IF_STMT,
    cindex.CursorKind.SWITCH_STMT,
    cindex.CursorKind.CASE_STMT,
    cindex.CursorKind.DEFAULT_STMT,
    cindex.CursorKind.FOR_STMT,
    cindex.CursorKind.WHILE_STMT,
    cindex.CursorKind.DO_STMT,
    cindex.CursorKind.GOTO_STMT,
    cindex.CursorKind.INDIRECT_GOTO_STMT,
    cindex.CursorKind.CONTINUE_STMT,
    cindex.CursorKind.BREAK_STMT,
}

LOOP_KINDS = {
    cindex.CursorKind.FOR_STMT,
    cindex.CursorKind.WHILE_STMT,
    cindex.CursorKind.DO_STMT,
}

CALL_KINDS = {
    cindex.CursorKind.CALL_EXPR,
    cindex.CursorKind.CXX_MEMBER_CALL_EXPR,
    cindex.CursorKind.CXX_OPERATOR_CALL_EXPR,
}

COMPARISON_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}
ASSIGNMENT_OPERATORS = {
    "=",
    "+=",
    "-=",
    "*=",
    "/=",
    "%=",
    "<<=",
    ">>=",
    "&=",
    "|=",
    "^=",
}
ARITHMETIC_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "multiplication",
    "/": "division",
}

COMPOUND_ARITHMETIC_MAP = {
    "+=": "+",
    "-=": "-",
    "*=": "*",
    "/=": "/",
}

FLOAT_TYPE_KINDS = {
    cindex.TypeKind.FLOAT,
    cindex.TypeKind.DOUBLE,
    cindex.TypeKind.LONGDOUBLE,
    cindex.TypeKind.FLOAT128,
    cindex.TypeKind.HALF,
}

INTEGER_TYPE_KINDS = {
    cindex.TypeKind.BOOL,
    cindex.TypeKind.CHAR_U,
    cindex.TypeKind.UCHAR,
    cindex.TypeKind.CHAR16,
    cindex.TypeKind.CHAR32,
    cindex.TypeKind.USHORT,
    cindex.TypeKind.UINT,
    cindex.TypeKind.ULONG,
    cindex.TypeKind.ULONGLONG,
    cindex.TypeKind.UINT128,
    cindex.TypeKind.SCHAR,
    cindex.TypeKind.WCHAR,
    cindex.TypeKind.SHORT,
    cindex.TypeKind.INT,
    cindex.TypeKind.LONG,
    cindex.TypeKind.LONGLONG,
    cindex.TypeKind.INT128,
}

ALLOCATION_FUNCTIONS = {"malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign", "_aligned_malloc"}


@dataclass
class FeatureAccumulator:
    """Holds all feature counters and helper aggregates."""

    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts.update(
            {
                "Total number of return statement": 0,
                "Total number of control statement": 0,
                "Total number of condition check instruction": 0,
                "Total number of function call instruction": 0,
                "Total number of functions": 0,
                "Total number of blocks": 0,
                "Total number of loops": 0,
                "Total number of loop operation": 0,
                "Total number of allocation instruction": 0,
                "Total number of load instruction": 0,
                "Total number of store instruction": 0,
                "Total number of multiplication (Float data type) operation": 0,
                "Total number of division (Float data type) instruction": 0,
                "Total number of addition (Float data type) instruction": 0,
                "Total number of subtraction (Float data type) instruction": 0,
                "Total number of multiplication (Integer data type) Instruction": 0,
                "Total number of division (Integer data type) instruction": 0,
                "Total number of addition (Integer data type) instruction": 0,
                "Total number of subtraction (Integer data type) instruction": 0,
                "Total number of instructions": 0,
                "Total number of float operation": 0,
                "Total number of integer operation": 0,
            }
        )

    def increment(self, key: str, amount: int = 1) -> None:
        self.counts[key] += amount

    def __getitem__(self, key: str) -> int:
        return self.counts[key]


class KernelFeatureExtractor:
    """Extracts kernel level program features using libclang."""

    def __init__(self, clang_library: Optional[Path] = None) -> None:
        if clang_library is not None:
            cindex.Config.set_library_file(str(clang_library))
        self.accumulator = FeatureAccumulator()

    def extract(self, source: Path, compile_args: Optional[List[str]] = None) -> Dict[str, int]:
        if not source.exists():
            raise FileNotFoundError(f"Source file '{source}' does not exist")

        self.accumulator = FeatureAccumulator()
        index = cindex.Index.create()
        translation_unit = index.parse(str(source), args=compile_args or [])
        self._visit(translation_unit.cursor, parent=None, inside_loop=False)
        self._finalize()
        return dict(self.accumulator.counts)

    def _visit(self, cursor: cindex.Cursor, parent: Optional[cindex.Cursor], inside_loop: bool) -> None:
        if cursor.kind.is_statement():
            self.accumulator.increment("Total number of instructions")

        if cursor.kind in LOOP_KINDS:
            self.accumulator.increment("Total number of loops")
            inside_loop = True

        if inside_loop and cursor.kind.is_statement():
            self.accumulator.increment("Total number of loop operation")

        if cursor.kind == cindex.CursorKind.RETURN_STMT:
            self.accumulator.increment("Total number of return statement")

        if cursor.kind in CONTROL_STATEMENT_KINDS:
            self.accumulator.increment("Total number of control statement")

        if cursor.kind == cindex.CursorKind.FUNCTION_DECL and cursor.is_definition():
            self.accumulator.increment("Total number of functions")

        if cursor.kind == cindex.CursorKind.COMPOUND_STMT:
            self.accumulator.increment("Total number of blocks")

        if cursor.kind in CALL_KINDS:
            self.accumulator.increment("Total number of function call instruction")
            self._maybe_count_allocation(cursor)

        if cursor.kind == cindex.CursorKind.CXX_NEW_EXPR:
            self.accumulator.increment("Total number of allocation instruction")

        if cursor.kind in {
            cindex.CursorKind.BINARY_OPERATOR,
            cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
        }:
            operator_spelling = self._extract_operator(cursor)
            self._maybe_record_condition(cursor, operator_spelling)
            self._maybe_record_store(cursor, operator_spelling)
            self._maybe_record_arithmetic(cursor, operator_spelling)

        if cursor.kind == cindex.CursorKind.DECL_REF_EXPR:
            if not self._is_store_target(cursor, parent) and not self._is_call_callee(cursor, parent):
                self.accumulator.increment("Total number of load instruction")

        if cursor.kind == cindex.CursorKind.MEMBER_REF_EXPR:
            if not self._is_store_target(cursor, parent) and not self._is_call_callee(cursor, parent):
                self.accumulator.increment("Total number of load instruction")

        if cursor.kind == cindex.CursorKind.ARRAY_SUBSCRIPT_EXPR and not self._is_store_target(cursor, parent):
            self.accumulator.increment("Total number of load instruction")

        for child in cursor.get_children():
            self._visit(child, parent=cursor, inside_loop=inside_loop)

    def _maybe_count_allocation(self, cursor: cindex.Cursor) -> None:
        referenced = cursor.referenced
        if referenced is not None and referenced.spelling in ALLOCATION_FUNCTIONS:
            self.accumulator.increment("Total number of allocation instruction")
            return

        if cursor.spelling in ALLOCATION_FUNCTIONS:
            self.accumulator.increment("Total number of allocation instruction")

    def _maybe_record_condition(self, cursor: cindex.Cursor, operator_spelling: Optional[str]) -> None:
        if operator_spelling in COMPARISON_OPERATORS:
            self.accumulator.increment("Total number of condition check instruction")

    def _maybe_record_store(self, cursor: cindex.Cursor, operator_spelling: Optional[str]) -> None:
        if operator_spelling in ASSIGNMENT_OPERATORS:
            self.accumulator.increment("Total number of store instruction")

    def _maybe_record_arithmetic(self, cursor: cindex.Cursor, operator_spelling: Optional[str]) -> None:
        if operator_spelling in COMPOUND_ARITHMETIC_MAP:
            operator_spelling = COMPOUND_ARITHMETIC_MAP[operator_spelling]

        if operator_spelling not in ARITHMETIC_OPERATORS:
            return

        operation = ARITHMETIC_OPERATORS[operator_spelling]
        result_type = cursor.result_type
        if result_type.kind in FLOAT_TYPE_KINDS:
            key = {
                "multiplication": "Total number of multiplication (Float data type) operation",
                "division": "Total number of division (Float data type) instruction",
                "addition": "Total number of addition (Float data type) instruction",
                "subtraction": "Total number of subtraction (Float data type) instruction",
            }[operation]
            self.accumulator.increment(key)
            self.accumulator.increment("Total number of float operation")
        elif result_type.kind in INTEGER_TYPE_KINDS:
            key = {
                "multiplication": "Total number of multiplication (Integer data type) Instruction",
                "division": "Total number of division (Integer data type) instruction",
                "addition": "Total number of addition (Integer data type) instruction",
                "subtraction": "Total number of subtraction (Integer data type) instruction",
            }[operation]
            self.accumulator.increment(key)
            self.accumulator.increment("Total number of integer operation")

    def _is_store_target(self, cursor: cindex.Cursor, parent: Optional[cindex.Cursor]) -> bool:
        if parent is None:
            return False
        if parent.kind not in {
            cindex.CursorKind.BINARY_OPERATOR,
            cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
        }:
            return False
        operator = self._extract_operator(parent)
        if operator not in ASSIGNMENT_OPERATORS:
            return False
        children = list(parent.get_children())
        return children and cursor == children[0]

    def _is_call_callee(self, cursor: cindex.Cursor, parent: Optional[cindex.Cursor]) -> bool:
        if parent is None or parent.kind not in CALL_KINDS:
            return False
        children = list(parent.get_children())
        return bool(children) and cursor == children[0]

    def _extract_operator(self, cursor: cindex.Cursor) -> Optional[str]:
        tokens = list(cursor.get_tokens())
        for token in tokens:
            if token.spelling in ARITHMETIC_OPERATORS or token.spelling in ASSIGNMENT_OPERATORS or token.spelling in COMPARISON_OPERATORS:
                return token.spelling
        return None

    def _finalize(self) -> None:
        float_total = (
            self.accumulator["Total number of multiplication (Float data type) operation"]
            + self.accumulator["Total number of division (Float data type) instruction"]
            + self.accumulator["Total number of addition (Float data type) instruction"]
            + self.accumulator["Total number of subtraction (Float data type) instruction"]
        )
        int_total = (
            self.accumulator["Total number of multiplication (Integer data type) Instruction"]
            + self.accumulator["Total number of division (Integer data type) instruction"]
            + self.accumulator["Total number of addition (Integer data type) instruction"]
            + self.accumulator["Total number of subtraction (Integer data type) instruction"]
        )
        self.accumulator.counts["Total number of float operation"] = float_total
        self.accumulator.counts["Total number of integer operation"] = int_total


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract kernel features from a C/C++ source file")
    parser.add_argument("source", type=Path, help="Path to the source file to analyse")
    parser.add_argument(
        "--clang-library",
        type=Path,
        default=None,
        help="Optional path to the libclang shared library",
    )
    parser.add_argument(
        "--compile-arg",
        action="append",
        default=None,
        help="Additional compilation arguments passed to libclang (can be repeated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where the CSV file will be created",
    )
    return parser.parse_args(argv)


def write_csv(output_dir: Path, source: Path, counts: Dict[str, int]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{source.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Feature", "Value"])
        for feature, value in counts.items():
            writer.writerow([feature, value])
    return csv_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    extractor = KernelFeatureExtractor(clang_library=args.clang_library)
    counts = extractor.extract(args.source, compile_args=args.compile_arg)
    csv_path = write_csv(args.output_dir, args.source, counts)
    print(f"Feature extraction completed. CSV saved to: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
