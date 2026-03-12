FC1_W = [
    [0.50, -0.25, 0.75, 0.10],
    [-0.40, 0.90, 0.30, -0.20],
    [0.60, 0.15, -0.55, 0.80],
]
FC1_B = [0.10, -0.20, 0.05]

FC2_W = [
    [0.70, -0.10, 0.20],
    [-0.30, 0.40, 0.60],
]
FC2_B = [0.03, -0.07]

FC3_W = [
    [1.10, -0.35],
]
FC3_B = [0.12]


def fully_connected(x, weights, bias):
    out = []
    for row, b in zip(weights, bias):
        total = sum(a * w for a, w in zip(x, row)) + b
        out.append(total)
    return out


def main():
    x = [1.0, -2.0, 0.5, 3.0]
    y1 = fully_connected(x, FC1_W, FC1_B)
    y2 = fully_connected(y1, FC2_W, FC2_B)
    y3 = fully_connected(y2, FC3_W, FC3_B)

    print("input =", x)
    print("fc1   =", [round(v, 6) for v in y1])
    print("fc2   =", [round(v, 6) for v in y2])
    print("fc3   =", [round(v, 6) for v in y3])


if __name__ == "__main__":
    main()
