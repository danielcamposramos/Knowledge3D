const SESSION_KEY = 'k3d-welcome-shown';

function markShown(): void {
  try {
    window.sessionStorage?.setItem(SESSION_KEY, '1');
  } catch {}
}

export function showWelcomeOverlay(): HTMLDivElement | null {
  try {
    if (window.sessionStorage?.getItem(SESSION_KEY) === '1') {
      return null;
    }
  } catch {}

  const overlay = document.createElement('div');
  overlay.className = 'k3d-welcome';
  Object.assign(overlay.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    background: 'rgba(0,0,0,0.7)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: '200',
    cursor: 'pointer',
  });

  overlay.innerHTML = `
    <div style="background: #1a1a2e; border: 1px solid #00ddff40; border-radius: 12px;
                padding: 32px 40px; max-width: 420px; color: white; font-family: monospace;
                text-align: center;">
      <h2 style="color: #00ddff; margin: 0 0 16px;">Knowledge3D House</h2>
      <p style="line-height: 1.6; margin: 0 0 20px; font-size: 14px; opacity: 0.8;">
        Navigate between rooms using <b>Arrow Keys</b> or <b>A/D</b>.<br>
        Jump to a room with <b>1-6</b> or press <b>H</b> for home.<br>
        <b>Click</b> objects to inspect them on the tablet.<br>
        <b>Drag</b> to orbit the camera.
      </p>
      <p style="font-size: 12px; opacity: 0.5;">Click anywhere or press any key to begin</p>
    </div>
  `;

  const dismiss = () => {
    if (!overlay.isConnected) return;
    overlay.remove();
    window.removeEventListener('keydown', onKeyDown, true);
  };
  const onKeyDown = () => dismiss();

  overlay.addEventListener('click', dismiss, { once: true });
  window.addEventListener('keydown', onKeyDown, true);
  (overlay as any).k3dDispose = dismiss;
  markShown();
  document.body.appendChild(overlay);
  return overlay;
}
