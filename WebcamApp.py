import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import os
from test import test, load_model  # Giả sử bạn đã có hàm load_model và test trong file test.py

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Anime Converter")

        # Thiết lập kích thước cửa sổ Tkinter để khớp với màn hình 10 inch (1280x800)
        self.root.geometry("1280x800")

        # Trạng thái
        self.captured_image = None
        self.display_image = None
        self.is_capturing = True

        # Load mô hình từ checkpoint một lần khi khởi tạo
        self.model_dir = "checkpoint/animeGan/Hayao/epoch=8-step=15012.ckpt"
        self.model = load_model(self.model_dir)

        # Thiết lập giao diện
        self.setup_ui()

        # Mở webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở webcam.")
            self.root.destroy()
            return

        # Thiết lập độ phân giải camera tương ứng với màn hình 10 inch
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Bắt đầu cập nhật khung hình từ webcam
        self.update_frame()

    def setup_ui(self):
        # Khung hiển thị ảnh, với kích thước lớn hơn để khớp với độ phân giải
        self.img_label = tk.Label(self.root)
        self.img_label.pack(expand=True, fill='both')

        # Tạo Frame chứa các nút
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        # Nút chụp ảnh và tự động chuyển thành anime
        self.capture_button = tk.Button(button_frame, text="Chụp & Chuyển thành Anime", command=self.capture_and_convert)
        self.capture_button.grid(row=0, column=0, padx=5, pady=5)

        # Nút hoàn tác để quay lại chụp ảnh tiếp
        self.undo_button = tk.Button(button_frame, text="Chụp ảnh mới", command=self.undo, state=tk.DISABLED)
        self.undo_button.grid(row=0, column=1, padx=5, pady=5)

    def update_frame(self):
        if self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                # Hiển thị ảnh bằng cách chuyển đổi từ BGR sang RGB (OpenCV -> PIL)
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Không cần resize vì đã đặt độ phân giải camera tương ứng với màn hình
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.img_label.imgtk = imgtk
                self.img_label.configure(image=imgtk)
        self.img_label.after(30, self.update_frame)  # Gọi lại hàm update_frame sau 30ms

    def capture_and_convert(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể chụp ảnh từ webcam")
            return

        # Chụp và lưu ảnh gốc (giữ nguyên định dạng BGR)
        self.captured_image = frame
        self.is_capturing = False

        original_image_name = "captured_image.jpg"
        image_input_path = os.path.join(r"D:\python-project\animeGanv2_pytorch\dataset\real", original_image_name)

        # Lưu ảnh BGR để sử dụng trong hàm test
        cv2.imwrite(image_input_path, self.captured_image)

        # Chuyển đổi thành ảnh anime
        self.convert_to_anime(image_input_path, original_image_name)

        # Kích hoạt nút hoàn tác
        self.undo_button.config(state=tk.NORMAL)

    def convert_to_anime(self, image_input_path, original_image_name):
        try:
            # Sử dụng mô hình đã load từ trước và truyền vào hàm test
            test(self.model, image_input_path, False)

            # Xác định đường dẫn của ảnh anime đã được lưu tự động
            anime_image_path = os.path.join(r"D:\python-project\animeGanv2_pytorch\results", original_image_name)

            # Kiểm tra xem ảnh anime có tồn tại không
            if os.path.exists(anime_image_path):
                # Đọc ảnh anime (có thể là ảnh BGR)
                anime_image = cv2.imread(anime_image_path)

                # Chuyển từ BGR sang RGB để hiển thị đúng màu
                anime_image_rgb = cv2.cvtColor(anime_image, cv2.COLOR_BGR2RGB)

                # Chuyển đổi từ numpy array sang PIL Image để hiển thị
                anime_image_pil = Image.fromarray(anime_image_rgb)
                imgtk_anime = ImageTk.PhotoImage(image=anime_image_pil)
                self.img_label.config(image=imgtk_anime)
                self.img_label.image = imgtk_anime
            else:
                messagebox.showerror("Lỗi", "Không thể tìm thấy ảnh anime sau khi chuyển đổi")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chuyển đổi ảnh thành anime: {e}")

    def undo(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.update_frame()
            self.undo_button.config(state=tk.DISABLED)

# Khởi tạo giao diện với Tkinter
root = tk.Tk()
app = WebcamApp(root)
# Khởi động vòng lặp giao diện
root.mainloop()

# Giải phóng camera khi đóng giao diện
if app.cap.isOpened():
    app.cap.release()
cv2.destroyAllWindows()
