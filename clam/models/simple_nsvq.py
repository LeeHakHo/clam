import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNSVQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        discarding_threshold: float = 0.01,
        initialization: str = "normal",  # 이젠 큰 의미 없지만 유지
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.discarding_threshold = discarding_threshold
        self.eps = eps

        # 초기에는 랜덤으로 두지만, 첫 forward때 데이터로 덮어씌울 것임
        self.codebook = nn.Parameter(torch.randn(codebook_size, dim))
        
        # [핵심 1] 초기화 여부 플래그 (Buffer로 저장하여 저장/로드 지원)
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.uint8))

        # Usage tracking
        self.register_buffer(
            "codebook_usage",
            torch.zeros(codebook_size, dtype=torch.long),
        )

    def _init_codebook(self, x_flat):
        """
        첫 배치의 데이터를 사용하여 코드북을 강제 초기화 (K-means++ 느낌)
        """
        with torch.no_grad():
            # 입력 데이터에서 랜덤하게 codebook_size만큼 뽑음
            n_data = x_flat.size(0)
            if n_data < self.codebook_size:
                # 데이터가 적으면 중복 허용해서 뽑음
                indices = torch.randint(0, n_data, (self.codebook_size,))
            else:
                # 데이터가 충분하면 중복 없이 뽑음
                indices = torch.randperm(n_data)[:self.codebook_size]
            
            # 선택된 데이터를 코드북에 복사
            selected_data = x_flat[indices].clone()
            
            # [핵심 2] L2 Normalize (선택사항이지만 수렴에 매우 도움됨)
            # 데이터와 코드북을 모두 구(Sphere) 위에 올리면 거리 계산이 훨씬 안정적임
            # 여기서는 데이터의 스케일만 맞춤
            self.codebook.data.copy_(selected_data)
            
            self.is_initialized.fill_(1)
            print(f"[SimpleNSVQ] Codebook initialized from data! (Shape: {self.codebook.shape})")

    def forward(self, x: torch.Tensor, codebook_training_only: bool = False):
        x_flat = x.reshape(-1, self.dim)
        
        # [핵심 1 적용] 학습 중이고 아직 초기화 안 됐으면, 지금 들어온 데이터로 초기화
        if self.training and self.is_initialized.item() == 0:
            self._init_codebook(x_flat)

        # ------------------------------------------------------------------
        # [핵심 3] L2 Normalization (옵션이지만 강력 추천)
        # 벡터의 크기(Magnitude) 차이로 인한 매핑 오류를 없애줍니다.
        # 데이터와 코드북을 모두 정규화해서 코사인 유사도 기반 매핑처럼 동작하게 함.
        # (원치 않으면 아래 두 줄 주석 처리)
        # x_norm = F.normalize(x_flat, dim=1)
        # codebook_norm = F.normalize(self.codebook, dim=1)
        # ------------------------------------------------------------------
        
        # 거리 계산 (x_flat과 self.codebook 사용)
        x_sq = (x_flat ** 2).sum(dim=1, keepdim=True)
        e_sq = (self.codebook ** 2).sum(dim=1)
        distances = x_sq - 2 * (x_flat @ self.codebook.t()) + e_sq.unsqueeze(0)

        indices = torch.argmin(distances, dim=1)
        codes = self.codebook[indices]

        # --- NSVQ Logic ---
        resid = x_flat - codes
        resid_norm = resid.norm(dim=1, keepdim=True)
        
        noise = torch.randn_like(x_flat)
        noise_norm = noise.norm(dim=1, keepdim=True)
        
        scaled_noise = (resid_norm / (noise_norm + self.eps)) * noise

        if codebook_training_only:
            quantized_flat = codes
        else:
            quantized_flat = x_flat + scaled_noise

        # --- Losses (지난번 수정사항 포함) ---
        commitment_loss = F.mse_loss(x_flat, codes.detach())
        codebook_loss = F.mse_loss(x_flat.detach(), codes)
        
        # Codebook Loss 비중을 좀 더 높여서(1.0) 데이터 쪽으로 강하게 당김
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # Usage Update
        if self.training:
            with torch.no_grad():
                self.codebook_usage.index_add_(
                    0, indices, torch.ones_like(indices, dtype=torch.long)
                )

        quantized = quantized_flat.view(*x.shape)
        indices = indices.view(*x.shape[:-1])

        return quantized, indices, vq_loss

    @torch.no_grad()
    def replace_unused_codebooks(self, num_batches: int):
        # 기존 로직 유지
        if num_batches <= 0: return
        usage_rate = self.codebook_usage.float() / float(num_batches)
        unused = torch.where(usage_rate < self.discarding_threshold)[0]
        used = torch.where(usage_rate >= self.discarding_threshold)[0]

        if used.numel() == 0:
            self.codebook.add_(self.eps * torch.randn_like(self.codebook))
        elif unused.numel() > 0:
            # 사용된 코드들 중에서 랜덤하게 뽑아서 죽은 코드 자리에 덮어쓰기
            # 약간의 노이즈를 섞어서 완전히 겹치지 않게 함
            used_codes = self.codebook[used]
            idx = torch.randint(0, used_codes.size(0), (unused.size(0),))
            self.codebook[unused] = used_codes[idx] + torch.randn_like(self.codebook[unused]) * 0.02
        
        self.codebook_usage.zero_()