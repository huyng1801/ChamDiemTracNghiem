import json
import pandas as pd

# Hàm để đọc tập tin JSON và trả về danh sách các bài thi
def load_exams_from_json(file_path):
    with open(file_path, "r") as f:
        exams = json.load(f)
    return exams

# Hàm để tạo một sheet trong file Excel từ một bài thi
def create_excel_sheet(exam, writer):
    df = pd.DataFrame(exam["questions"])
    sheet_name = f"{exam['exam_id']}"
    df.to_excel(writer, sheet_name=sheet_name, index=False)

# Đọc các bài thi từ tập tin JSON
exams = load_exams_from_json("exams.json")

# Tạo một writer cho file Excel
with pd.ExcelWriter("exams.xlsx") as writer:
    # Tạo một sheet cho mỗi bài thi
    for exam in exams:
        create_excel_sheet(exam, writer)

print("File Excel đã được tạo thành công.")
