echo "Downloading GloVe word vectors (used by the evaluator)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing
rm -rf glove
unzip glove.zip
rm glove.zip
echo "Downloading done!"
