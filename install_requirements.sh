#!/bin/bash

# requirements.txt ファイルの存在を確認
if [[ ! -f requirements.txt ]]; then
  echo "requirements.txt ファイルが見つかりません。"
  exit 1
fi

# requirements.txt ファイルを一行ずつ読み込んで処理
while IFS= read -r line; do
  # 空行やコメント行（#で始まる行）は無視
  if [[ -z "$line" || "$line" == \#* ]]; then
    continue
  fi

  # rye add コマンドを実行
  echo "rye add $line"
  rye add "$line"
done < requirements.txt

echo "すべてのパッケージがインストールされました。"
