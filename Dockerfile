# 1. ベースイメージの選択
# PyTorch, CUDA 11.8, cuDNN 8 がプリインストールされた公式イメージを使用します。
# これにより、CUDAのバージョン互換性の問題を解決します。
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2. 作業ディレクトリの設定
# コンテナ内での作業場所を /app に設定します。
WORKDIR /app

# 3. 依存関係のインストール (キャッシュの活用)
# まず requirements.txt だけをコピーしてライブラリをインストールします。
# こうすることで、ソースコードの変更時に毎回ライブラリを再インストールするのを防ぎます。
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. プロジェクトファイルのコピー
# 残りのプロジェクトファイルを作業ディレクトリにコピーします。
COPY . .

# 5. デフォルトのコマンド (コンテナ起動時に実行される)
# コンテナが起動し続けるように、bashを起動しておきます。
CMD ["bash"]
