import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import csv
from torch.utils.data import Dataset, DataLoader


class MinecraftHouseDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512, device='cpu'):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        self.device = device

        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Пропускаем пустые строки
                    continue
                try:
                    # Пробуем разные разделители
                    if '\t' in line:
                        description, structure = line.split('\t')
                    elif '    ' in line:  # Четыре пробела
                        description, structure = line.split('    ')
                    else:
                        raise ValueError(f"Не удалось разделить строку: {line}")

                    self.data.append((description, structure))
                except ValueError as e:
                    print(f"Ошибка в строке {line_num}: {e}")
                    print(f"Проблемная строка: {line}")
                    continue  # Пропускаем проблемную строку и продолжаем

        if not self.data:
            raise ValueError("Не удалось загрузить данные. Проверьте формат файла.")

        print(f"Загружено {len(self.data)} пар данных.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        description, structure = self.data[idx]

        input_encoding = self.tokenizer(description,
                                        max_length=self.max_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")

        target_encoding = self.tokenizer(structure,
                                         max_length=self.max_length,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")

        return {
            "input_ids": input_encoding.input_ids.flatten().to(self.device),
            "attention_mask": input_encoding.attention_mask.flatten().to(self.device),
            "labels": target_encoding.input_ids.flatten().to(self.device),
        }

# Шедвро fine-tune модели.
def train_model(model, tokenizer, train_file, val_file, epochs=20, batch_size=4, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    model.to(device)

    train_dataset = MinecraftHouseDataset(train_file, tokenizer, device=device)
    val_dataset = MinecraftHouseDataset(val_file, tokenizer, device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average training loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")

    return model


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
                print(f"Предупреждение: Пропуск недопустимого формата блока: {block}")
                continue
            x, y, z, block_type = parts
            blocks.append((int(x), int(y), int(z), block_type))
        except ValueError as e:
            print(f"Некорректный формат блока: {block}. Ошибка: {e}")

    if not blocks:
        raise ValueError("Нет допустимых блоков!")

    return blocks


def save_to_csv(blocks, filename="house_blocks.csv"):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Z", "Block"])
        writer.writerows(blocks)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Загрузка модели и токенизатора
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    # Пути к файлам данных
    train_file = "train_data.txt"
    val_file = "val_data.txt"

    # Обучение модели
    print("Начало обучения модели...")
    try:
        trained_model = train_model(model, tokenizer, train_file, val_file)
        print("Обучение завершено.")

        # Сохранение обученной модели
        trained_model.save_pretrained("fine_tuned_minecraft_t5")
        tokenizer.save_pretrained("fine_tuned_minecraft_t5")
        print("Модель сохранена.")

        # Генерация дома
        while True:
            user_prompt = input("Опишите дом, который вы хотите построить (или 'выход' для завершения): ")
            if user_prompt.lower() == 'выход':
                break

            print("Генерация структуры дома...")
            house_structure = generate_house(trained_model, tokenizer, user_prompt)

            print("Обработка структуры...")
            blocks = parse_house_structure(house_structure)

            print("Сохранение структуры в CSV...")
            save_to_csv(blocks)

            print("Структура дома сохранена в файле 'house_blocks.csv'")
            print("Вы можете использовать этот файл с плагином Minecraft для построения дома.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Пожалуйста, проверьте формат файлов данных и попробуйте снова.")


if __name__ == "__main__":
    main()