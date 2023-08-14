import numpy as np
import cv2



def display(board):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("- - - + - - - + - - -")
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(board[i, j], end=" ")
        print()


# Check if 'num' can be placed at position (row, col) in the board
# by verifying row, column, and 3x3 box constraints.
def is_valid(board, row, col, num):
    # Check row and column
    if num in board[row, :] or num in board[:, col]:
        return False

    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    if num in board[box_row:box_row + 3, box_col:box_col + 3]:
        return False

    return True


# Solve the sudoku puzzle using a backtracking algorithm.
def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row, col] == 0:
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        board[row, col] = num
                        if solve_sudoku(board):
                            return True
                        board[row, col] = 0
                return False
    return True


def extract_sudoku(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sudoku_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                sudoku_contour = approx
                break

    if sudoku_contour is None:
        print("Sudoku grid not found.")
        return

    dest_corners = np.array([[0, 0], [0, 450], [450, 450], [450, 0]], dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(sudoku_contour.astype(np.float32), dest_corners)
    warped = cv2.warpPerspective(image, transform_matrix, (450, 450))

    grid = np.zeros((9, 9), dtype=np.uint8)
    cell_size = warped.shape[0] // 9
    for i in range(9):
        for j in range(9):
            cell = warped[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            #add paddleocr recognition here
            if results:
                digit = results[0][1]
                if digit.isnumeric():
                    grid[i, j] = int(digit)

    return grid


puzzle = np.array([
    [8, 0, 0, 4, 0, 6, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 7, 0, 0, 0, 6, 0, 0],
    [0, 9, 0, 2, 0, 8, 0, 6, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 9, 0, 3, 0, 1, 0],
    [0, 0, 3, 0, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 1, 0, 2, 0, 0, 8]])

# if solve_sudoku(puzzle):
#     print("Solution:")
#     display(puzzle)
# else:
#     print("No solution exists.")

print(extract_sudoku(r"C:\Users\krist\OneDrive\Legion\Scripts\PySudoku\Sudoku.jpg"))
