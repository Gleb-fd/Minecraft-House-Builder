import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def generate_house(model, tokenizer, prompt, max_length=200):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated output: {decoded_output}")  # Debug print
    return decoded_output


def parse_house_structure(structure):
    blocks = []
    for block in structure.split():
        try:
            parts = block.split(',')
            if len(parts) != 4:
                print(f"Warning: Skipping invalid block format: {block}")
                continue
            x, y, z, block_type = parts
            blocks.append((int(x), int(y), int(z), block_type))
        except ValueError as e:
            print(f"Error parsing block: {block}. Error: {e}")

    if not blocks:
        raise ValueError("нет валидных сгенерированных блоков!")

    return blocks

def save_to_csv(blocks, filename="house_blocks.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z", "Block"])
        writer.writerows(blocks)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = T5ForConditionalGeneration.from_pretrained("fine_tuned_minecraft_t5").to(device)
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_minecraft_t5")

# Генерация дома
while True:
    try:
        user_prompt = input("Опишите дом, который вы хотите построить (или 'выход' для завершения): ")
        if user_prompt.lower() == 'выход':
            break
        print("Генерация структуры дома...")
        house_structure = generate_house(model, tokenizer, user_prompt)
        print("Обработка структуры...")
        blocks = parse_house_structure(house_structure)
        print("Сохранение структуры в CSV...")
        save_to_csv(blocks)
        print("Структура дома сохранена в файле 'house_blocks.csv'")
        print("Вы можете использовать этот файл с плагином Minecraft для построения дома.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Пожалуйста, проверьте формат файлов данных и попробуйте снова.")