# zh_text_classification
- 项目包含以下内容
  - 使用单向 rnn 进行文本分类；
  - 使用双向 birnn 进行文本分类；
  - 使用 cnn 进行文本分类 参考论文([Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf))；
  
## 数据集
   数据集格式 : label words 其中 label 是标签，words 是分词后的单词词语，label 和 words 之间以 tab 键分隔，words 之间以 ，隔开

## 代码说明
- data_helper.py 数据预处理
- rnn_model.py 单向 rnn 模型
- bilstm_model.py 双向 rnn 模型
- cnn_model.py cnn 模型
- run.py 运行程序
