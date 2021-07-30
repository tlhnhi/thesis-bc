# Demo Khóa luận tốt nghiệp - K17 - HCMUS

Đây là ứng dụng web minh họa khả năng hoạt động của kết quả được trình bày trong khóa luận đề tài _Áp dụng kỹ thuật học sâu hỗ trợ phát hiện và phân loại tế bào máu theo hình dạng_

## Sinh viên thực hiện

| Student ID | Name             |
| ---------- | ---------------- |
| 1753081    | Trần Lê Hồng Nhi |
| 1753083    | Nguyễn Hưng Phát |

## Thư viện hỗ trợ

- **Pytorch**

(_Cài đặt theo hướng dẫn của Pytorch [tại đây](https://pytorch.org/get-started/locally/)_)

- **Detectron2**

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

- **Flask**

```
pip install Flask
```

## Cách sử dụng

- Clone repo này về máy và di chuyển vào thư mục repo
- Tải file `R50_FPN_3x.pth` tại đây (_**chưa cập nhập url**_) và chuyển vào thư mục chính của repo

```
- thesis-bc\
|---- statis\
|---- templates\
|---- READEME.md
|---- demo_app.py
|---- R50_FPN_3x.pth
```

- Export biến môi trường `FLASK_APP`
  - Với Bash
  ```
  export FLASK_APP=demo_app.py
  ```
  - Với CMD
  ```
  set FLASK_APP=demo_app.py
  ```
  - Với Powershell
  ```
  $env:FLASK_APP = "demo_app.py"
  ```
- Chạy server

```
flask run
```

- Sau khi chờ model được tải lên, server sẽ chạy tại http://127.0.0.1:5000/
