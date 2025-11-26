# Các bước triển khai
## Task 1: Tải & tiền xử lý dữ liệu
- Tải dữ liệu: Sử dụng datasets.load_dataset("lhoestq/conll2003") để lấy tập CoNLL-2003 chuẩn, gồm 3 split: train, validation, test.
- Chuyển nhãn số sang string: Nhãn NER ban đầu là số, ánh xạ sang tên nhãn "B-PER", "I-PER", ...
- Xây dựng từ điển:
  - word_to_ix: ánh xạ từ token sang chỉ số integer; thêm <PAD> và <UNK>.
  - tag_to_ix: ánh xạ nhãn sang chỉ số; thêm <PAD> cho padding.
- Thêm <PAD> và <UNK> để xử lý batch padding và từ chưa thấy.

## Task 2: Dataset & DataLoader
- Lớp NERDataset: kế thừa torch.utils.data.Dataset; chuyển câu và nhãn sang tensor indices.
- collate_fn: dùng pad_sequence để đệm các batch về cùng độ dài.
- DataLoader: tạo batch với shuffle=True cho train, shuffle=False cho validation/test.

## Task 3: Xây dựng Mô hình
- Embedding layer: chuyển từ chỉ số sang vector 100 chiều.
- Bi-LSTM:
  - 1 lớp, bidirectional, 128 hidden units.
  - Nhận chuỗi embedding và encode context trái/phải.
- Linear layer: ánh xạ hidden state → số lượng nhãn.
- Forward pass: output [batch, seq_len, num_classes] dùng cho CrossEntropyLoss.

## Task 4: Huấn luyện mô hình
- Optimizer: Adam với learning rate 1e-3.
- Loss function: nn.CrossEntropyLoss(ignore_index=PAD_TAG_IDX) để bỏ qua padding token.
- Vòng lặp train:
  - model.train(), zero_grad()
  - Forward pass -> tính logits
  - Reshape logits & targets để phù hợp CrossEntropyLoss ([batch*seq_len, num_classes])
  - Backward pass → optimizer.step()
- Epochs: 3, in loss trung bình.

## Task 5: Đánh giá Mô hình
- Hàm evaluate:
  - model.eval(), tắt gradient
  - torch.argmax trên logits cuối cùng → nhãn dự đoán cho từng token
  - Bỏ padding token khi tính accuracy
  - Tính F1, precision, recall bằng seqeval
- Hàm predict_sentence:
  - Chuyển câu string sang indices, forward pass → argmax → cặp (word, predicted_tag)


- Độ chính xác trên tập validation: 
• Ví dụ dự đoán câu mới:

3

– Câu: “VNU University is located in Hanoi”
– Dự đoán: ...

