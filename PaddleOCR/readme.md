## Ghi chú về PaddleOCR

PaddleOCR là một thư viện OCR mạnh mẽ, dễ sử dụng và hỗ trợ nhiều ngôn ngữ.  
Trong quá trình thử nghiệm, mô hình cho kết quả **xác định vị trí vùng văn bản khá chính xác**.  

Tuy nhiên, độ chính xác về **nội dung nhận dạng** chưa cao, ngay cả khi giá trị `confidence` được trả về ở mức lớn.  
Điều này có nghĩa là mô hình có thể "tự tin" với kết quả, nhưng nội dung thực tế vẫn sai lệch.  