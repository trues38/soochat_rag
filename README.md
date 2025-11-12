# SooChat-RAG Solution (Dify + Qdrant + Supabase)

## 1. 개요
Dify 워크플로우를 기반으로 하는 소상공인 마케팅 및 채팅 자동화 솔루션입니다.
- **AI Engine:** Dify
- **Vector DB:** Qdrant
- **Primary DB:** PostgreSQL (Supabase 역할)

## 2. 설치 및 실행 (Docker 필수)
1. DIFY_API_KEY, QDRANT_API_KEY 등 모든 환경 변수를 설정합니다.
2. 아래 명령어로 모든 컨테이너를 실행합니다.
   docker-compose up -d

## 3. 핵심 파일 구조
- docker-compose.yml: 전체 서비스 정의
- backend/: Dify API를 호출하는 래퍼(Wrapper) 로직
- data/: Public RAG 및 초기 지식 데이터

