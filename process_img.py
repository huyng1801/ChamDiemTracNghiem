import os
import pandas as pd
import imutils
import numpy as np
import cv2
from math import ceil
from model import CNN_Model
from collections import defaultdict
import json

def get_x(s):
    return s[1][0]


def get_y(s):
    return s[1][1]


def get_h(s):
    return s[1][3]


def get_x_ver1(s):
    s = cv2.boundingRect(s)
    return s[0] * s[1]

def crop_image(img):
    # Chuyển đổi ảnh từ không gian màu BGR sang GRAY để áp dụng thuật toán canny edge detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Loại bỏ nhiễu bằng cách làm mờ ảnh
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Áp dụng thuật toán canny edge detection
    img_canny = cv2.Canny(blurred, 100, 200)

    # Tìm các đường viền
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    ans_blocks = []
    x_old, y_old, w_old, h_old = 0, 0, 0, 0

    # Đảm bảo rằng ít nhất một đường viền đã được tìm thấy
    if len(cnts) > 0:
        # Sắp xếp các đường viền theo kích thước của chúng theo thứ tự giảm dần
        cnts = sorted(cnts, key=get_x_ver1)

        # Lặp qua các đường viền đã được sắp xếp
        for i, c in enumerate(cnts):
            x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

            if w_curr * h_curr > 100000:
                # Kiểm tra đè lên các đường viền
                check_xy_min = x_curr * y_curr - x_old * y_old
                check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                # Nếu danh sách khối câu trả lời là trống
                if len(ans_blocks) == 0:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # Cập nhật tọa độ (x, y) và (chiều cao, chiều rộng) của các đường viền đã thêm
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr
                elif check_xy_min > 20000 and check_xy_max > 20000:
                    ans_blocks.append(
                        (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr], [x_curr, y_curr, w_curr, h_curr]))
                    # Cập nhật tọa độ (x, y) và (chiều cao, chiều rộng) của các đường viền đã thêm
                    x_old = x_curr
                    y_old = y_curr
                    w_old = w_curr
                    h_old = h_curr

        # Sắp xếp ans_blocks theo tọa độ x
        sorted_ans_blocks = sorted(ans_blocks, key=get_x)
        return sorted_ans_blocks

def process_ans_blocks(ans_blocks):
    """
        Hàm này xử lý 2 khối câu trả lời và trả về một danh sách câu trả lời có độ dài là 200 lựa chọn ô nhiễm
        :param ans_blocks: một danh sách bao gồm 2 phần tử, mỗi phần tử có định dạng [image, [x, y, w, h]]
    """
    list_answers = []

    # Lặp qua mỗi khối câu trả lời
    for ans_block in ans_blocks:
        ans_block_img = np.array(ans_block[0])

        offset1 = ceil(ans_block_img.shape[0] / 6)
        # Lặp qua mỗi hộp trong khối câu trả lời
        for i in range(6):
            box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
            height_box = box_img.shape[0]

            box_img = box_img[14:height_box - 14, :]
            offset2 = ceil(box_img.shape[0] / 5)

            # lặp qua mỗi dòng trong một hộp
            for j in range(5):
                list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])

    return list_answers



def process_list_ans(list_answers):
    list_choices = []
    offset = 44
    start = 32

    for answer_img in list_answers:
        for i in range(4):
            bubble_choice = answer_img[:, start + i * offset:start + (i + 1) * offset]
            bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
            bubble_choice = bubble_choice.reshape((28, 28, 1))
            list_choices.append(bubble_choice)

    if len(list_choices) != 480:
        raise ValueError("Length of list_choices must be 480")
    return list_choices


def map_answer(idx):
    if idx % 4 == 0:
        answer_circle = "A"
    elif idx % 4 == 1:
        answer_circle = "B"
    elif idx % 4 == 2:
        answer_circle = "C"
    else:
        answer_circle = "D"
    return answer_circle


def get_answers(list_answers):
    results = defaultdict(list)
    model = CNN_Model('weight.h5').build_model(rt=True)
    list_answers = np.array(list_answers)
    scores = model.predict_on_batch(list_answers / 255.0)
    for idx, score in enumerate(scores):
        question = idx // 4

        # score [unchoiced_cf, choiced_cf]
        if score[1] > 0.9:  # Điểm tự tin của câu trả lời được chọn > 0.9
            chosen_answer = map_answer(idx)
            results[question + 1].append(chosen_answer)
        
    # Kiểm tra nếu có bất kỳ câu hỏi nào không có câu trả lời được chọn
    for question_id in range(1, len(scores) // 4 + 1):
        if question_id not in results:
            results[question_id] = []  # Gán một danh sách rỗng cho các câu hỏi không có câu trả lời được chọn
    
    # Sắp xếp kết quả theo id câu hỏi
    sorted_results = {k: results[k] for k in sorted(results)}
    
    return sorted_results



def detect_exam_code_cnn(image):
    # Tải hoặc khởi tạo model CNN (chỉ cần thực hiện một lần)
    model = CNN_Model('weight.h5').build_model(rt=True)

    # Chuyển ảnh sang ảnh grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Áp dụng phân ngưỡng
    _, thresh = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
    # Tìm các đường viền
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Lọc các đường viền dựa trên diện tích
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]


    # Tìm đường viền có diện tích nhỏ nhất (giả định là mã đề)
    min_area_contour = None
    min_area = float('inf')
    for contour in valid_contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            min_area = area
            min_area_contour = contour

    # Kiểm tra nếu tìm thấy đường viền phù hợp
    if min_area_contour is None:
        print("Không tìm thấy đường viền mã đề!")
        return None

    x, y, w, h = cv2.boundingRect(min_area_contour)
    
    exam_code_area = image[y:y+h, x:x+w]
    cv2.imwrite("exam_code.jpg", exam_code_area)
    # Chia vùng mã đề thành 3 cột và 10 ô mỗi cột
    # Lấy chiều cao và chiều rộng từ exam_code_area
    h, w = exam_code_area.shape[:2]  # Thay đổi ở đây
    
    # Chia vùng mã đề thành 3 cột và 10 ô mỗi cột
    column_width = w // 3
    cell_height = h // 10
    
    exam_code = ""
    for col_idx in range(3):
        for row_idx in range(10):
            
            # Tính toán vị trí của ô trong mã đề
            cell_x = col_idx * column_width
            cell_y = row_idx * cell_height
            cell_img = exam_code_area[cell_y:cell_y+cell_height, cell_x:cell_x+column_width]
            
            if cell_img.size == 0:
                print(f"Ô ({col_idx}, {row_idx}) rỗng!")
                continue  # Bỏ qua ô rỗng
            # Chuyển đổi cell_img sang grayscale
            cell_img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

            # Thresholding với ảnh grayscale
            processed_cell = cv2.threshold(cell_img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            processed_cell = cv2.resize(processed_cell, (28, 28), cv2.INTER_AREA)
            processed_cell = processed_cell.reshape((28, 28, 1))

            # Dự đoán với model
            prediction = model.predict(np.array([processed_cell]) / 255.0, verbose=False)
            # Giả sử model trả về xác suất ô được tô ở prediction[0][1]
            if prediction[0][1] > 0.9:
                digit = row_idx  # Giá trị số từ 0 đến 9
                exam_code += str(digit)


    return exam_code

def detect_id_number(image):
    model = CNN_Model('weight.h5').build_model(rt=True)
    # Chuyển ảnh sang ảnh grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Áp dụng phân ngưỡng
    _, thresh = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
    # Tìm các đường viền
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Lọc các đường viền dựa trên diện tích
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    # Sắp xếp đường viền theo diện tích tăng dần
    valid_contours.sort(key=cv2.contourArea)

    # Kiểm tra nếu có đủ đường viền
    if len(valid_contours) < 2:
        print("Không tìm thấy đủ đường viền cho mã đề và số báo danh!")
        return None

    # Lấy đường viền có diện tích nhỏ thứ hai (vị trí thứ 1 trong danh sách)
    second_smallest_contour = valid_contours[1]
    x, y, w, h = cv2.boundingRect(second_smallest_contour)

    id_number_area = image[y:y+h, x:x+w]
    cv2.imwrite("id_number.jpg", id_number_area)
    # Chia vùng mã đề thành 3 cột và 10 ô mỗi cột
    h, w = id_number_area.shape[:2]  # Thay đổi ở đây
    
    
    # Chia vùng số báo danh thành 6 cột và 10 dòng
    column_width = w // 6
    cell_height = h // 10

    id_number = ""
    for col_idx in range(6):
        for row_idx in range(10):
            cell_x = col_idx * column_width
            cell_y = row_idx * cell_height
            cell_img = id_number_area[cell_y:cell_y + cell_height, cell_x:cell_x + column_width]
          
            if cell_img.size == 0:
                print(f"Ô ({col_idx}, {row_idx}) rỗng!")
                continue  # Bỏ qua ô rỗng
            
            cell_img_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            # Thresholding với ảnh grayscale
            processed_cell = cv2.threshold(cell_img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            processed_cell = cv2.resize(processed_cell, (28, 28), cv2.INTER_AREA)
            processed_cell = processed_cell.reshape((28, 28, 1))
            # Dự đoán với model
            prediction = model.predict(np.array([processed_cell]) / 255.0, verbose=False)
            # Giả sử model trả về xác suất ô được tô ở prediction[0][1]
            if prediction[0][1] > 0.9:
                digit = row_idx  # Giá trị số từ 0 đến 9
                id_number += str(digit)

    return id_number

def load_exams_from_excel(file_path):
    excel_data = pd.ExcelFile(file_path)
    exams_data = []

    for sheet_name in excel_data.sheet_names:
        df = pd.read_excel(excel_data, sheet_name=sheet_name)
        exam_id = int(sheet_name)
        questions = df.to_dict('records')
        exam = {"exam_id": exam_id, "questions": questions}
        exams_data.append(exam)

    return exams_data
def score_and_print_answers(received_answers, exam_id):
    exams_data = load_exams_from_excel("exams.xlsx")

    exam = next((exam for exam in exams_data if exam["exam_id"] == int(exam_id)), None)

    exam = next((exam for exam in exams_data if exam["exam_id"] == int(exam_id)), None)

    if exam is None:
        print(f"Không tìm thấy đề thi có mã {exam_id}.")
        return None

    correct_answers = {question["question_id"]: question["correct_answer"] for question in exam["questions"]}

    score = 0


    total_questions = len(correct_answers)
    questions_printed = 0 


    print("+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
    print("|" + "Câu hỏi".center(8) + "|" + "Lựa chọn".center(10) + "|" + "Đáp án".center(10) + "|" + "Kết quả".center(10) + "|")
    
    for question_id, choices in received_answers.items():
        if questions_printed >= len(correct_answers):
            break
        if len(choices) == 0:
                correct_choice = correct_answers.get(question_id)
                print("+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
                print(f"|{str(question_id).center(8)}|{''.center(10)}|{correct_choice.center(10)}|{'Sai'.center(10)}|")
        elif len(choices) == 1:
            correct_choice = correct_answers.get(question_id)
            result = "Đúng" if correct_choice == choices[0] else "Sai"
            score += 1 if correct_choice == choices[0] else 0
            print("+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
            print(f"|{str(question_id).center(8)}|{choices[0].center(10)}|{correct_choice.center(10)}|{result.center(10)}|")      
        else:
            correct_choice = correct_answers.get(question_id)
            choices_str = ", ".join(choices)
            print("+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
            print(f"|{str(question_id).center(8)}|{choices_str.center(10)}|{correct_choice.center(10)}|{'Sai'.center(10)}|")      
        questions_printed += 1 
    print("+" + "-" * 8 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 10 + "+")
    print(f"Kết quả: {score}/{total_questions}")
    print(f"Điểm số: {round(10 / total_questions * score, 2)}")
    
def get_student_name(student_id, file_path='student.json'):

    with open(file_path, 'r', encoding='utf-8') as file:
        student_data = json.load(file)
    
    return student_data.get(student_id)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')
    
if __name__ == '__main__': 
    img = cv2.imread('203.png')
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)

    id_number = detect_id_number(img)
    exam_code = detect_exam_code_cnn(img)
    student_name = get_student_name(id_number)
    clear_console()
    print("Mã đề:", exam_code)
    print("Số báo danh:", id_number)
    print("Họ và tên:", student_name)
    score_and_print_answers(answers, exam_code)


