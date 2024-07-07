def generate_house():
    house = []
    house.append("Build a small wooden house with windows")
    for z in range(4):
        for y in range(4):
            for x in range(3):
                if y == 1 and (x == 1 or x == 2) and (z == 1 or z == 2):
                    block = "GLASS_PANE"
                elif y == 1 and x == 1 and (z == 0 or z == 3):
                    block = "AIR"
                elif y == 3:
                    block = "OAK_STAIRS"
                else:
                    block = "OAK_PLANKS"
                house.append(f"{x},{y},{z},{block}")
    return " ".join(house)

houses = [generate_house() for _ in range(100)]

for house in houses:
    print(house)




