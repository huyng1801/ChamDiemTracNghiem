import json

# Hàm để tạo đối tượng câu hỏi
def create_question(question_id, text, answers, correct_answer):
    return {
        "question_id": question_id,
        "text": text,
        "answers": answers,
        "correct_answer": correct_answer
    }

# Hàm để tạo một bài thi với ID cho trước
def create_exam(exam_id):
    exam = {"exam_id": exam_id, "questions": []}
    for i in range(1, 51):
        question_text = f"Câu hỏi {i} của đề {exam_id}"
        answers = {"A": "A", "B": "B", "C": "C", "D": "D"}
        correct_answer = "A" if i % 4 == 0 else "B" if i % 4 == 1 else "C" if i % 4 == 2 else "D"
        exam["questions"].append(create_question(i, question_text, answers, correct_answer))
    return exam

# Tạo các bài thi từ 201 đến 208
exams = []
for i in range(201, 209):
    exams.append(create_exam(i))

# Lưu các bài thi vào một tập tin JSON
with open("exams.json", "w") as f:
    json.dump(exams, f, indent=2, ensure_ascii=False)

print("Các bài thi đã được tạo và lưu vào 'exams.json'.")
