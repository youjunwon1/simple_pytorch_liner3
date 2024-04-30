Without using an encoder model in PyTorch, 

a text prediction (just using book text) 

with only embeddings, 3 linear layers, and activation functions


def __init__(self):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(len(g_vocab2text), embedding_dim)  #  임베딩 레이어

        self.elu = nn.ELU(alpha=1.0)
        self.dropout = nn.Dropout(drop_out_rate)
        self.output_layer1 = nn.Linear(embedding_dim, (embedding_dim*2))  #1번층
        self.output_layer2 = nn.Linear((embedding_dim*2), embedding_dim) #2번층
        self.output_layer3 = nn.Linear(embedding_dim, len(g_vocab2text)) #3번층



# 모델 훈련
for epoch in range(num_epochs):
    total_loss = 0



    # 배치로 하지 않고 문장의 길이로 처리  (어짜피 문장의 길이가 길어지면 처리 시간도 길어질테니까 ..

    for i in range(quotient-1):   # 남는 문장은 무시  ~ 450단어를 40단어로 나누면 25 :  25단어 무시..
                                  # 마지막 문장 -1은 모델의 예측에 넣어야 해서 뺌

        index = i * sentence_length         #0,40,80,120  (sentence_length가  40인경
        index2 = (sentence_length-1) +(i*sentence_length)  #39, 79,119,

        target_index = i * sentence_length +  sentence_length        #40,80,  (sentence_length가  40인경
        target_index2 = (sentence_length-1) +(i*sentence_length) + sentence_length  #79, 119,


        #모델에 들어갈 데이터와  값 비교할 데이터 추출
        input_subtensor = inputs_split_data_indexed_tensor[:, index:index2]  # 인덱스는 0부터 시작하므로 40번 인덱스부터 79번 인덱스까지 가져옴
        target_subtensor = inputs_split_data_indexed_tensor[:, target_index:target_index2]


        # 모델의 예측 수행
        #입력에는 첫번째 40 단어
        #출력에는 두번째 40 단어
        #print(input_subtensor)
        output_tensor = model(input_subtensor)


        target_subtensor = target_subtensor.squeeze(0)
        output_tensor = output_tensor.squeeze(0)
        #print(output_tensor.size())
        #print("target_subtensor",target_subtensor.size())


        loss = criterion(output_tensor.view(-1, len(g_vocab2text)), target_subtensor)


        total_loss += loss.item()

        # 옵티마이저 초기화 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # 각 에포크별 손실 출력
    if epoch % 50 == 0:

        print(f'Epoch {epoch + 1}, Loss: {total_loss :.4f}')

        end_time = time.time()  # 종료 시간 기록
        elapsed_time = end_time - start_time  # 총 학습 시간 계산
        print(f"학습 시간 {elapsed_time:.4f} 초")


        #_, predicted_index = torch.sort(output_tensor, dim=-1, descending=True)

        _, predicted_index = torch.max(output_tensor, dim=-1)

        #print("predicted_index   max ", predicted_index)

        predicted_word = ""
        for i in range(len(predicted_index)):
            predicted_word = predicted_word + " "+  g_vocab2text[predicted_index[i].item()]
        print("predicted_word",predicted_word)

        #가장 확률 높은 단어 추출
        #print("가장 확률 높은 단어 추출", output_tensor)

        #####################################



        #print(target_subtensor)  #원래의 글
        #print(target_subtensor[10].item())

        original_target_word = ""
        for i in range(sentence_length-1):
            original_target_word = original_target_word + " "+  g_vocab2text[target_subtensor[i].item()]
        print("original_target_word",original_target_word)


        ########################################
        #질문 하면 대답 듣는 걸로 한번 해보기
        #대홍수에 밀려내려가는 소리
        tempdata = "대홍수에 밀려내려가 는 소리"

        prepared_data = prepareData(tempdata)
        tensored_data = tokenize_and_index(prepared_data, g_vocab)
        tensored_data = torch.tensor(tensored_data).to(device)

        shape = tensored_data.shape
        tensored_data = tensored_data.reshape((shape[1], shape[0]))  #행,열로 모양 변경


        #print("tensored_data.size",tensored_data.size())



        outputed = model(tensored_data).to(device)
        outputed = outputed.squeeze(0)
        #outputed.view(-1, len(g_vocab2text))

        softmax_output = F.softmax(outputed, dim=-1)

        #print("output_tensor", output_tensor)
        # 결과 출력
        #print("softmax_output", softmax_output)

        #맥스 값이 이상하게 도 질문의 단어수에 맞춰서 출력된다  (질문단어 5개 답변 단어 5개 왜 ??)
        _, predicted_index = torch.max(softmax_output, dim=-1)
        print("predicted_index   max ", predicted_index)

        predicted_word = ""
        for i in range(len(predicted_index)):
            predicted_word = predicted_word + " "+  g_vocab2text[predicted_index[i].item()]

        print("대홍수에 밀려내려가는 소리에 대한 답변 ",predicted_word)

