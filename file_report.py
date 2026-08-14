"""递归统计当前工作目录内各文件的行数和大小。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# 不扫描版本控制数据和 Python 自动生成的缓存文件。
IGNORED_DIRECTORIES = {".git", "__pycache__", ".venv"}


@dataclass(frozen=True)
class FileStat:
    """单个文件的统计结果。"""

    path: str
    lines: int
    size: int


def count_lines(path: Path) -> int:
    """按字节流统计行数，避免文件编码或二进制内容导致读取失败。"""
    with path.open("rb") as file:
        content = file.read()

    if not content:
        return 0
    return content.count(b"\n") + (0 if content.endswith(b"\n") else 1)


def collect_file_stats(root: Path) -> tuple[list[FileStat], list[str]]:
    """收集 root 下的文件数据，并保留无法读取文件的错误信息。"""
    stats: list[FileStat] = []
    errors: list[str] = []

    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in IGNORED_DIRECTORIES
        ]
        directory_path = Path(directory)

        for filename in filenames:
            file_path = directory_path / filename
            try:
                stats.append(
                    FileStat(
                        path=file_path.relative_to(root).as_posix(),
                        lines=count_lines(file_path),
                        size=file_path.stat().st_size,
                    )
                )
            except OSError as error:
                errors.append(f"{file_path.relative_to(root)}: {error}")

    return sorted(stats, key=lambda item: item.path), errors


def format_size(size: int) -> str:
    """将字节数格式化为易读单位。"""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    return f"{size} B"


def print_report(stats: list[FileStat], errors: list[str]) -> None:
    """将统计结果以对齐表格输出到终端。"""
    path_width = max(len("文件路径"), *(len(item.path) for item in stats))
    line_width = max(len("行数"), *(len(str(item.lines)) for item in stats))
    size_width = max(len("大小"), *(len(format_size(item.size)) for item in stats))

    divider = "-" * (path_width + line_width + size_width + 10)
    print("文件统计报告")
    print(divider)
    print(f"{'文件路径':<{path_width}} | {'行数':>{line_width}} | {'大小':>{size_width}}")
    print(divider)

    for item in stats:
        print(
            f"{item.path:<{path_width}} | {item.lines:>{line_width}} | "
            f"{format_size(item.size):>{size_width}}"
        )

    print(divider)
    print(
        f"汇总：{len(stats)} 个文件，{sum(item.lines for item in stats)} 行，"
        f"{format_size(sum(item.size for item in stats))}"
    )

    if errors:
        print("\n以下文件无法读取：")
        for error in errors:
            print(f"- {error}")


def main() -> None:
    """扫描当前工作目录并输出报告。"""
    root = Path.cwd()
    stats, errors = collect_file_stats(root)
    print_report(stats, errors)


if __name__ == "__main__":
    main()
