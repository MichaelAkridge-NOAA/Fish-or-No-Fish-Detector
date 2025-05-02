# Yes-Fish / No-Fish Arcade – Pygame + YOLOv8  (v 3.3)
# =============================================================================
#  NEW IN 3.3
#  • Main-menu item “Confidence: x.xx” lets you adjust the YOLO confidence
#    threshold live with ← / → keys (0.05 – 0.95 in 0.05 steps).
#  • Threshold is stored in Config.CONF_THRESHOLD and used on every inference.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, sys, json, random, time, pygame
from dataclasses import dataclass
from pygame import Rect, Surface
from ultralytics import YOLO

# ───────────────────────── Configuration ─────────────────────────
from pathlib import Path
BASE = Path(__file__).parent                     # .../.devcontainer/yes-no-fish

@dataclass
class Config:
    # paths now built off BASE instead of bare strings
    IMAGE_DIR        = BASE / "images"
    MODEL_PATH       = BASE / "model" / "yolov11n_fish_trained.pt"
    LEADERBOARD_FILE = BASE / "leaderboard.json"
    LOGO_PATH        = BASE / "assets" / "logo.png"
    SND_CORRECT      = BASE / "assets" / "correct.wav"
    SND_WRONG        = BASE / "assets" / "wrong.wav"

    TIME_LIMIT       = 60          # seconds per round
    CONF_THRESHOLD   = 0.6        # ◄ adjustable in menu
    FPS              = 60
    SLIDE_MS         = 350         # slide duration (ms)

    SCREEN_W, SCREEN_H = 1280, 800
    BG   = (18, 20, 24)
    FG   = (235, 235, 235)
    ACC  = (46, 196, 182)
    BOX  = (0, 255, 0)
    PFISH    = (0, 220, 0)
    PNOFISH  = (220, 50, 50)

    BTN_IDLE_A   = 140
    BTN_FLASH_A  = 220
    FONT_NAME    = "freesansbold.ttf"

CFG = Config()

# menu index map for clarity
IDX_START, IDX_SFX, IDX_CONF, IDX_HOW, IDX_LB, IDX_QUIT = range(6)
MENU = ["Start Game",
        "Toggle SFX",
        "Confidence",      # label filled dynamically
        "How-to-Play",
        "Leaderboard",
        "Quit"]

# ───────────────────────── Leaderboard ──────────────────────────
class Leaderboard:
    def __init__(self, fp: str):
        self.fp = fp
        self.scores = self._load()

    def _load(self):
        if os.path.exists(self.fp):
            try:   return json.load(open(self.fp, "r", encoding="utf-8"))
            except Exception:   return []
        return []

    def _save(self):  json.dump(self.scores, open(self.fp, "w", encoding="utf-8"), indent=2)

    def add(self, name: str, reviewed: int, acc: float):
        self.scores.append({"name": name, "reviewed": reviewed, "accuracy": acc})
        self.scores.sort(key=lambda x: (-x["reviewed"], -x["accuracy"]))
        self._save()

    def top(self, k=10):  return self.scores[:k]

# ───────────────────────── Game Round ───────────────────────────
class FishRound:
    def __init__(self, sc, ck, fonts, lb: Leaderboard, sfx_on: bool):
        self.sc, self.ck = sc, ck
        self.fbig, self.fsm = fonts
        self.lb, self.sfx_on = lb, sfx_on
        self._init_sounds(); self._init_yolo(); self._load_images(); self._make_buttons()

    # ---------- init helpers ----------
    def _init_sounds(self):
        self.snd_ok = self.snd_ng = None
        if not self.sfx_on: return
        if os.path.exists(CFG.SND_CORRECT): self.snd_ok = pygame.mixer.Sound(CFG.SND_CORRECT)
        if os.path.exists(CFG.SND_WRONG):   self.snd_ng = pygame.mixer.Sound(CFG.SND_WRONG)

    def _init_yolo(self):
        try:  self.model = YOLO(CFG.MODEL_PATH)
        except Exception:
            print("! YOLO weights missing – random mode", file=sys.stderr); self.model = None

    def _load_images(self):
        ims = [os.path.join(CFG.IMAGE_DIR, f) for f in os.listdir(CFG.IMAGE_DIR)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        if not ims: raise RuntimeError(f"No images in {CFG.IMAGE_DIR}")
        random.shuffle(ims); self.images = ims

    def _make_buttons(self):
        def make(txt, col, x_frac):
            t = self.fbig.render(txt, True, (255,255,255)); pad = 40
            surf = Surface((t.get_width()+pad, t.get_height()+pad), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*col, CFG.BTN_IDLE_A), surf.get_rect(), border_radius=20)
            surf.blit(t, t.get_rect(center=surf.get_rect().center))
            rect = surf.get_rect(center=(CFG.SCREEN_W*x_frac, CFG.SCREEN_H*0.82))
            return surf, rect
        self.btn_yes, self.r_yes = make("YES", CFG.PFISH,   0.75)
        self.btn_no , self.r_no  = make("NO" , CFG.PNOFISH, 0.25)
        self.flash_ms = 0; self.flash_side = "YES"

    # ---------- detection ----------
    def detect(self, path):
        if self.model is None: return random.random() > 0.5, []
        r = self.model(path, verbose=False, conf=CFG.CONF_THRESHOLD)[0]
        boxes = [Rect(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                      int(b.xyxy[0][2]-b.xyxy[0][0]), int(b.xyxy[0][3]-b.xyxy[0][1]))
                 for b in r.boxes if float(b.conf) >= CFG.CONF_THRESHOLD]
        return bool(boxes), boxes

    def scale_img(self, surf: Surface):
        f = min((CFG.SCREEN_W*0.8)/surf.get_width(), (CFG.SCREEN_H*0.6)/surf.get_height())
        return pygame.transform.smoothscale(surf, (int(surf.get_width()*f), int(surf.get_height()*f))), f

    def load_img(self, idx):
        p = self.images[idx]; raw = pygame.image.load(p).convert(); pred, bxs = self.detect(p)
        surf, s = self.scale_img(raw)
        rect = surf.get_rect(center=(CFG.SCREEN_W//2, CFG.SCREEN_H//2-60))
        sb = [Rect(int(b.x*s), int(b.y*s), int(b.w*s), int(b.h*s)) for b in bxs]
        return surf, rect, pred, sb

    # ---------- buttons ----------
    def _draw_buttons(self):
        a_yes = CFG.BTN_FLASH_A if self.flash_ms and self.flash_side=="YES" else CFG.BTN_IDLE_A
        a_no  = CFG.BTN_FLASH_A if self.flash_ms and self.flash_side=="NO"  else CFG.BTN_IDLE_A
        self.btn_yes.set_alpha(a_yes); self.btn_no.set_alpha(a_no)
        self.sc.blit(self.btn_yes, self.r_yes); self.sc.blit(self.btn_no, self.r_no)

    # ---------- main loop ----------
    def play(self):
        cur_idx = 0
        cur_surf, cur_rect, cur_pred, cur_boxes = self.load_img(cur_idx)

        # placeholders for “next” card (needed for nonlocal)
        nxt_surf: Surface|None = None; nxt_rect: Rect|None=None
        nxt_pred: bool|None = None;   nxt_boxes: list[Rect] = []

        next_ready = False; reviewed = correct = 0
        sliding = False; slide_dir = 1; slide_start = 0
        centre_x = CFG.SCREEN_W//2; start_ms = pygame.time.get_ticks()

        def start_slide(dir_right: bool):
            nonlocal sliding, slide_dir, slide_start, next_ready
            nonlocal nxt_surf, nxt_rect, nxt_pred, nxt_boxes
            sliding = True; slide_dir = 1 if dir_right else -1
            slide_start = pygame.time.get_ticks()
            ni = (cur_idx + 1) % len(self.images)
            nxt_surf, nxt_rect, nxt_pred, nxt_boxes = self.load_img(ni)
            nxt_rect.centerx += -slide_dir * CFG.SCREEN_W
            next_ready = True

        running = True
        while running:
            dt = self.ck.tick(CFG.FPS); now = pygame.time.get_ticks()
            rem = max(0, CFG.TIME_LIMIT - (now - start_ms)//1000)
            running = rem > 0
            if self.flash_ms: self.flash_ms = max(0, self.flash_ms - dt)

            # events
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN and not sliding and ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    user_yes = (ev.key==pygame.K_RIGHT); good = (user_yes==cur_pred)
                    correct += good; reviewed += 1
                    if self.sfx_on and (snd := self.snd_ok if good else self.snd_ng): snd.play()
                    self.flash_side = "YES" if user_yes else "NO"; self.flash_ms = 200
                    start_slide(user_yes)

            # slide animation
            if sliding:
                t = (now - slide_start) / CFG.SLIDE_MS
                if t >= 1:
                    cur_idx = (cur_idx + 1) % len(self.images)
                    cur_surf, cur_rect, cur_pred, cur_boxes = nxt_surf, nxt_rect, nxt_pred, nxt_boxes
                    sliding = False; next_ready=False
                else:
                    off = slide_dir * t * CFG.SCREEN_W
                    cur_rect.centerx = centre_x + off
                    nxt_rect.centerx = centre_x + off - slide_dir*CFG.SCREEN_W

            # draw
            self.sc.fill(CFG.BG)
            self.sc.blit(cur_surf, cur_rect)
            for b in cur_boxes:
                pygame.draw.rect(self.sc, CFG.BOX, b.move(cur_rect.topleft), 3)
            if sliding and next_ready:
                self.sc.blit(nxt_surf, nxt_rect)
                for b in nxt_boxes:
                    pygame.draw.rect(self.sc, CFG.BOX, b.move(nxt_rect.topleft), 3)

            self._draw_buttons()
            self._txt(f"Model: {'Fish' if cur_pred else 'No Fish'}",
                      (20,20), CFG.PFISH if cur_pred else CFG.PNOFISH)
            self._txt(f"Time: {rem}s", (20,60))
            self._txt(f"Reviewed: {reviewed}  Correct: {correct}", (20,100))
            pygame.display.flip()

        acc = round(correct/reviewed, 3) if reviewed else 0
        self.lb.add(self._prompt_initials(), reviewed, acc)

    # ---------- tiny helpers ----------
    def _txt(self, m, pos, col=None):
        sf = self.fsm.render(m, True, col or CFG.FG); self.sc.blit(sf, sf.get_rect(topleft=pos))

    def _prompt_initials(self):
        name=""; entering=True
        while entering:
            self.ck.tick(30)
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_RETURN and name: entering=False
                    elif ev.key==pygame.K_BACKSPACE: name=name[:-1]
                    elif len(name)<10 and ev.unicode.isprintable(): name+=ev.unicode.upper()
            self.sc.fill(CFG.BG)
            p=self.fbig.render("Enter initials: "+name, True, CFG.FG)
            self.sc.blit(p, p.get_rect(center=(CFG.SCREEN_W//2, CFG.SCREEN_H//2))); pygame.display.flip()
        return name

# ──────────────────────── Main App & Menus ────────────────────────
class App:
    def __init__(self):
        pygame.init(); pygame.mixer.init()
        self.sc = pygame.display.set_mode((CFG.SCREEN_W, CFG.SCREEN_H))
        pygame.display.set_caption("Yes-Fish / No-Fish Arcade")
        self.ck = pygame.time.Clock()
        self.fbig = pygame.font.Font(CFG.FONT_NAME, 54); self.fsm = pygame.font.Font(CFG.FONT_NAME, 26)
        self.lb = Leaderboard(CFG.LEADERBOARD_FILE); self.logo = self._logo(); self.sfx_on = True

    def _logo(self):
        if os.path.exists(CFG.LOGO_PATH):
            raw = pygame.image.load(CFG.LOGO_PATH).convert_alpha()
            h = int(CFG.SCREEN_H*0.4); w = int(raw.get_width()*h/raw.get_height())
            return pygame.transform.smoothscale(raw, (w,h))
        return None

    def run(self):
        self._splash()
        while True:
            sel = self._menu()
            if   sel == IDX_START: FishRound(self.sc, self.ck, (self.fbig,self.fsm), self.lb, self.sfx_on).play()
            elif sel == IDX_SFX:   self.sfx_on = not self.sfx_on
            elif sel == IDX_HOW:   self._dialog([
                                    "• Confirm as many images as possible in 60 s.",
                                    "• Model pre-labels Fish / No-Fish.",
                                    "• →  YES (Fish) ←  NO (No-Fish).",
                                    "• Use menu item to tune confidence threshold.",
                                    "", "Press any key to return." ])
            elif sel == IDX_LB:    self._show_leaderboard()
            else:                  pygame.quit(); sys.exit()

    # ---------- splash ----------
    def _splash(self):
        if not self.logo: return
        t0=time.time(); wait=True
        while wait:
            self.ck.tick(60)
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN: wait=False
            if time.time()-t0>2: wait=False
            self.sc.fill(CFG.BG); self.sc.blit(self.logo, self.logo.get_rect(center=(CFG.SCREEN_W//2, CFG.SCREEN_H//2)))
            pygame.display.flip()

    # ---------- menu ----------
    def _menu(self):
        sel = 0
        while True:
            self.ck.tick(30)
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_UP:   sel=(sel-1)%len(MENU)
                    elif ev.key==pygame.K_DOWN: sel=(sel+1)%len(MENU)
                    elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and sel==IDX_CONF:
                        step = -0.05 if ev.key==pygame.K_LEFT else 0.05
                        CFG.CONF_THRESHOLD = min(0.95, max(0.05, round(CFG.CONF_THRESHOLD+step,2)))
                    elif ev.key in (pygame.K_RETURN, pygame.K_RIGHT) and sel!=IDX_CONF:
                        return sel

            self.sc.fill(CFG.BG)
            self.sc.blit(self.fbig.render("Yes-Fish / No-Fish", True, CFG.ACC),
                         (CFG.SCREEN_W//2 - 210, 140))

            for i,item in enumerate(MENU):
                if i==IDX_SFX:
                    label = f"SFX: {'ON' if self.sfx_on else 'OFF'}"
                elif i==IDX_CONF:
                    label = f"Confidence: {CFG.CONF_THRESHOLD:.2f}"
                else:
                    label = item
                col = CFG.ACC if i==sel else CFG.FG
                s = self.fsm.render(label, True, col)
                self.sc.blit(s, s.get_rect(center=(CFG.SCREEN_W//2, 300+i*40)))
            pygame.display.flip()

    def _dialog(self, lines):
        wait=True
        while wait:
            self.ck.tick(30)
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN: wait=False
            self.sc.fill(CFG.BG)
            for i,l in enumerate(lines):
                f=self.fbig if i==0 else self.fsm
                s=f.render(l, True, CFG.FG)
                self.sc.blit(s, s.get_rect(center=(CFG.SCREEN_W//2, 140+i*50)))
            pygame.display.flip()

    def _show_leaderboard(self):
        lines = ["Leaderboard"] + [
            f"{i+1}. {e['name']:10s}  {e['reviewed']:3d} imgs  |  {e['accuracy']*100:5.1f}%"
            for i,e in enumerate(self.lb.top())
        ] + ["", "Press any key to return."]
        self._dialog(lines)

# ──────────────────────────────── Entry ────────────────────────────────
if __name__ == "__main__":
    App().run()
