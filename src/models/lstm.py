"""
PURPOSE:
  Predicts future turbulence parameters (r0, tau0, wind) from history.
  Enables PREDICTIVE AO — pre-position DM before turbulence arrives.
  Eliminates 10-20% Strehl loss from temporal lag.
PHYSICS BASIS that used to build the lstm network:
  Taylor's frozen turbulence hypothesis:
    Atmosphere moves as a rigid screen at wind speed v.
    Future wavefront = current wavefront shifted by v × Δt.
    This makes future states PREDICTABLE from current measurements.
INPUT : (r0(t-19:t), tau0(t-19:t), vx(t-19:t), vy(t-19:t)) — last 20 frames
OUTPUT: (r0(t+1:t+5), tau0, vx, vy) — next 5 frames predicted
"""
