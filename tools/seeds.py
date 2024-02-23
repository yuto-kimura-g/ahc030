N = 1500
with open("seeds.txt", "w") as f:
    for i in range(N + 1):
        f.write(f"{i}\n")
