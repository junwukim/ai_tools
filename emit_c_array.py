import argparse
import os


def to_identifier(name):
    result = []
    for ch in name:
        if ch.isalnum():
            result.append(ch.lower())
        else:
            result.append('_')
    ident = ''.join(result).strip('_')
    while '__' in ident:
        ident = ident.replace('__', '_')
    return ident or 'model_data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--symbol", default=None)
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        data = f.read()

    base_symbol = args.symbol or to_identifier(os.path.splitext(os.path.basename(args.input_path))[0])
    array_name = f"g_{base_symbol}"
    len_name = f"g_{base_symbol}_len"
    guard = f"{base_symbol.upper()}_DATA_H_"

    lines = [
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        f"alignas(16) const unsigned char {array_name}[] = {{",
    ]

    for offset in range(0, len(data), 12):
        chunk = data[offset:offset + 12]
        lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")

    lines += [
        "};",
        f"const unsigned int {len_name} = {len(data)};",
        "",
        f"#endif  // {guard}",
        "",
    ]

    with open(args.output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
