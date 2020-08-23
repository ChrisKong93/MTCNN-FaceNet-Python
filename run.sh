#!/bin/sh
#检查程序是否在运行
if [ "-r" = "$1" ]; then
    processcount=`ps -fe|grep face_server |grep -v grep|wc -l`
    echo '检查程序是否运行 ${processcount}'
    if [ $processcount -gt 0 ]; then
      echo '杀死正在运行的程序'
      kill -s 9 `ps -aux | grep "face_server" | grep -v grep| awk '{print $2}'`
    fi
    sleep 3s
fi

#!/bin/sh
#检查程序是否在运行

aipid=`ps -ef|grep face_server |grep -v grep|wc -l`
if [ $aipid -gt 0 ]; then
  echo "AI程序已经在运行了${aipid}"
else
  #当前文件所在目录
  CURRENT_DIR=$(cd `dirname $0`; pwd)
  echo "进入目录:${CURRENT_DIR}"
  cd $CURRENT_DIR
  CUDA_VISIBLE_DEVICES=1 nohup python3 -u face_server.py >> log 2>&1 &
  echo "开启人脸识别程序成功"
fi
