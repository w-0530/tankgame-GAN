#!/usr/bin/env python3
import sys
import os

def main():
    print("坦克游戏AI训练器")
    print("=" * 30)
    print("1. 训练AI")
    print("2. 测试AI")
    print("3. 退出")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        print("开始训练AI...")
        os.system("python train.py")
    elif choice == "2":
        print("开始测试AI...")
        os.system("python train.py test")
    elif choice == "3":
        print("再见！")
        sys.exit(0)
    else:
        print("无效选择！")
        main()

if __name__ == "__main__":
    main()