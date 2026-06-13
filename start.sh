#!/usr/bin/env bash
echo ""
echo " ================================================================"
echo "   CLS Finder  --  Compact Localized State Calculator"
echo " ================================================================"
echo ""
echo " 로컬 HTTP 서버를 시작합니다..."
echo " 서버를 중지하려면 Ctrl+C 를 누르세요."
echo ""

if ! command -v python3 &>/dev/null; then
    echo " [ERROR] python3를 찾을 수 없습니다."
    exit 1
fi

# Open browser (macOS / Linux)
if command -v open &>/dev/null; then
    (sleep 2 && open http://localhost:8765/web/) &
elif command -v xdg-open &>/dev/null; then
    (sleep 2 && xdg-open http://localhost:8765/web/) &
fi

echo " 서버 주소: http://localhost:8765/web/"
echo ""
python3 -m http.server 8765
