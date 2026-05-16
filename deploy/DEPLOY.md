# Deploy lên Oracle Cloud Always Free (ARM) — Docker Compose

Mục tiêu: chạy Traffic RAG (Qdrant + FastAPI + Next.js + Nginx) trên 1 VM Oracle Free Tier ARM Ampere, truy cập qua `http://<IP>`.

---

## 1. Đăng ký Oracle Cloud Free Tier

1. Vào https://cloud.oracle.com → "Start for free".
2. Verify bằng credit/debit card (không charge — chỉ verify identity).
3. Chọn **home region** gần VN: **Singapore (ap-singapore-1)** hoặc **Tokyo (ap-tokyo-1)**.
4. Sau khi tạo tài khoản, vào **Compute → Instances → Create Instance**.

### Thông số instance

| Field | Value |
|---|---|
| Name | `traffic-rag` |
| Image | **Canonical Ubuntu 22.04** (ARM) |
| Shape | **VM.Standard.A1.Flex** (Ampere) |
| OCPU | **4** |
| Memory | **24 GB** |
| Boot volume | **50 GB** (mặc định) |
| SSH | Generate new SSH key → **lưu private key `.pem`** |

> Nếu báo "Out of capacity" → đổi sang region khác (Tokyo, Osaka, Mumbai) hoặc thử lại sau vài giờ. Đây là phần khó nhất của Oracle Free Tier.

### Mở port 80 ở Oracle firewall

Sau khi instance chạy:
1. Click vào instance → **Subnet** → **Default Security List**.
2. **Add Ingress Rule**:
   - Source CIDR: `0.0.0.0/0`
   - IP Protocol: `TCP`
   - Destination port: `80`

---

## 2. SSH vào server

```bash
chmod 600 ~/Downloads/ssh-key-xxx.pem
ssh -i ~/Downloads/ssh-key-xxx.pem ubuntu@<PUBLIC_IP>
```

(Lấy `<PUBLIC_IP>` ở trang chi tiết instance.)

---

## 3. Upload project lên server

Trên **máy local**, từ thư mục cha của `GitHub1/`:

```bash
rsync -avz --progress \
  --exclude='GitHub1/traffic_rag/Data/raw' \
  --exclude='GitHub1/traffic_rag/Data/cleaned' \
  --exclude='GitHub1/traffic_rag/Data/chroma_bench' \
  --exclude='GitHub1/traffic_rag/Data/semantic_chunks' \
  --exclude='GitHub1/traffic_rag/research' \
  --exclude='GitHub1/traffic_rag/logs' \
  --exclude='GitHub1/traffic_rag/.git' \
  --exclude='GitHub1/traffic_rag/app/nextjs-app/node_modules' \
  --exclude='GitHub1/traffic_rag/app/nextjs-app/.next' \
  --exclude='GitHub1/Retrieval_Information_System_With_Agentic_RAG-main' \
  -e "ssh -i ~/Downloads/ssh-key-xxx.pem" \
  GitHub1/ ubuntu@<PUBLIC_IP>:~/GitHub1/
```

Project sẽ ở `/home/ubuntu/GitHub1/` trên server. Tổng dung lượng ~100MB (đã include `qdrant_storage/` 39MB và `all_chunks.jsonl` 4.5MB).

> Nếu repo đã push lên GitHub: chỉ cần `git clone <repo-url> ~/GitHub1` rồi `scp` riêng `.env` và `qdrant_storage/`.

---

## 4. Chạy setup script

SSH vào server, rồi:

```bash
cd ~/GitHub1/traffic_rag/deploy
bash setup-oracle.sh
```

Script sẽ:
- Cài Docker + Compose v2
- Mở iptables port 80 (Oracle Ubuntu chặn mặc định ngoài SSH)
- Tạo swap 2GB
- Generate `.env.frontend` với IP server + NEXTAUTH_SECRET random
- Build + start tất cả containers

**Lần đầu build ~8-12 phút** (download base image ARM + cài Python deps + pre-download embedding model 1.1GB).

Sau khi script chạy xong, log ra:
```
==> Deploy complete.
    Frontend:  http://<PUBLIC_IP>
```

---

## 5. Kiểm tra

```bash
# Trên server:
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f backend  # Ctrl+C để thoát
```

Trên browser:
- `http://<PUBLIC_IP>` — UI chat
- `http://<PUBLIC_IP>/admin` — HITL approval (đăng nhập Google bắt buộc, hoặc bỏ auth — xem §6)

Hỏi thử: *"Vượt đèn đỏ ô tô bị phạt bao nhiêu?"*

---

## 6. Cấu hình Google OAuth (tùy chọn)

`/admin` yêu cầu đăng nhập Google. Nếu chưa setup OAuth:

1. https://console.cloud.google.com → APIs & Services → Credentials.
2. Create OAuth Client ID → Web application.
3. Authorized redirect URI: `http://<PUBLIC_IP>/api/auth/callback/google`.
4. Copy Client ID + Secret vào `~/GitHub1/traffic_rag/deploy/.env.frontend`.
5. Restart: `docker compose -f docker-compose.prod.yml restart frontend`.

Nếu chỉ demo: để trống và truy cập `/` (chat không cần auth).

---

## 7. Lệnh thường dùng

```bash
cd ~/GitHub1/traffic_rag/deploy

# Restart 1 service
docker compose -f docker-compose.prod.yml restart backend

# Xem log
docker compose -f docker-compose.prod.yml logs -f --tail=100

# Rebuild sau khi sửa code
docker compose -f docker-compose.prod.yml up -d --build

# Tắt toàn bộ
docker compose -f docker-compose.prod.yml down

# Tắt + xóa volumes (sẽ mất frontend DB + HF cache, NHƯNG qdrant_storage là bind mount nên giữ nguyên)
docker compose -f docker-compose.prod.yml down -v
```

---

## 8. Gắn domain + HTTPS (sau, tùy chọn)

Khi đã mua domain (`.id.vn`, `.xyz`...):

1. DNS: trỏ A record `@` → `<PUBLIC_IP>`.
2. Trên server:
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   ```
3. Tạm dừng nginx container:
   ```bash
   docker compose -f docker-compose.prod.yml stop nginx
   ```
4. Lấy cert:
   ```bash
   sudo certbot certonly --standalone -d yourdomain.com
   ```
5. Sửa `nginx.conf` thêm block listen 443 + `ssl_certificate` trỏ về `/etc/letsencrypt/live/yourdomain.com/`.
6. Sửa `docker-compose.prod.yml`: mount `/etc/letsencrypt:/etc/letsencrypt:ro` vào nginx, expose thêm port 443.
7. Sửa `.env.frontend`: `NEXTAUTH_URL=https://yourdomain.com`.
8. Mở port 443 trên Oracle Security List + iptables.
9. `docker compose -f docker-compose.prod.yml up -d`.

---

## 9. Troubleshooting

| Triệu chứng | Xử lý |
|---|---|
| `http://<IP>` timeout | Check Oracle Security List ingress port 80 + iptables (script đã mở, nhưng verify: `sudo iptables -L INPUT -n | grep 80`) |
| Backend OOM | Đổi container memory limit hoặc tăng swap. Free Tier ARM 24GB không gặp. |
| Qdrant collection trống | `docker compose exec backend python -m source.indexing.indexer --recreate` |
| Build chậm/lỗi | `docker compose build --no-cache backend` |
| Gemini quota hết | Đổi `API_KEY` trong `.env` (xem comment đa key trong file). Restart backend. |
| Embedding download chậm | First build pull ~1.1GB từ HuggingFace, kiên nhẫn. Cache lại trong volume `hf_cache`. |

---

## Cost summary

- **Oracle Free Tier**: $0/tháng (vĩnh viễn nếu tài khoản hoạt động).
- **Gemini API**: free tier ~15 req/min, đủ demo.
- **Tavily**: 1000 search/tháng free.
- **Domain (optional)**: $1-5/năm.

**Tổng**: $0/tháng nếu không mua domain.
