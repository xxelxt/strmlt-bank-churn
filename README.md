## 📤 Upload file lớn lên GitHub?

<details>
<summary>Nhấn vào đây</summary>

1. Tải Git LFS về.

2. Di chuyển đến folder repo, mở `cmd` và chạy lệnh:
```
git lfs install
```

3. Thêm file vào Git LFS:
```
git lfs track <your_file>
git add .gitattributes
```

4. Commit và push như khi dùng Git:
```
git add <your_file>
git commit -m "<your_description>"
git push origin main
```

5. File lớn không tự fetch về khi clone repo đâu, nên là:
```
git lfs fetch
```

</details>
