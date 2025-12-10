
# -*- coding: utf-8 -*-
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

V_TARGET_CM3 = 330.0

# --- Geometry functions ---
def cylinder_given_r(r_cm: float):
    if r_cm <= 0:
        return math.nan, math.nan
    h_cm = V_TARGET_CM3 / (math.pi * r_cm**2)
    A_cm2 = 2 * math.pi * r_cm * h_cm + 2 * math.pi * (r_cm**2)
    return h_cm, A_cm2

def frustum_given_rt_rb(rt_cm: float, rb_cm: float):
    if rt_cm <= 0 or rb_cm <= 0:
        return math.nan, math.nan
    denom = math.pi * (rb_cm**2 + rb_cm*rt_cm + rt_cm**2)
    if denom <= 0:
        return math.nan, math.nan
    h_cm = 3 * V_TARGET_CM3 / denom
    s_cm = math.sqrt((rb_cm - rt_cm)**2 + h_cm**2)
    A_side_cm2 = math.pi * (rb_cm + rt_cm) * s_cm
    A_caps_cm2 = math.pi * (rb_cm**2) + math.pi * (rt_cm**2)
    A_total_cm2 = A_side_cm2 + A_caps_cm2
    return h_cm, A_total_cm2

def cuboid_given_LW(L_cm: float, W_cm: float):
    if L_cm <= 0 or W_cm <= 0:
        return math.nan, math.nan
    H_cm = V_TARGET_CM3 / (L_cm * W_cm)
    A_cm2 = 2 * (L_cm * W_cm + L_cm * H_cm + W_cm * H_cm)
    return H_cm, A_cm2

def cube_fixed_volume():
    a_cm = V_TARGET_CM3 ** (1.0 / 3.0)
    A_cm2 = 6.0 * (a_cm ** 2)
    return a_cm, A_cm2

def square_pyramid_given_b(b_cm: float):
    if b_cm <= 0:
        return math.nan, math.nan
    h_cm = 3.0 * V_TARGET_CM3 / (b_cm ** 2)
    l_cm = math.sqrt((b_cm / 2.0) ** 2 + h_cm ** 2)
    A_lat_cm2 = 2.0 * b_cm * l_cm
    A_total_cm2 = b_cm ** 2 + A_lat_cm2
    return h_cm, A_total_cm2

st.set_page_config(page_title='330 mL Container Surface Area (Web - Matplotlib)', layout='wide')
st.title('330 mL Container Surface Area Calculator')

shape = st.radio('選擇形狀 Shape', ['Cylinder', 'Frustum', 'Cuboid', 'Cube', 'Square Pyramid'], horizontal=True)

col1, col2 = st.columns([1,1])

r = rt = rb = L = W = H = a = b = h = None

with col1:
    if shape == 'Cylinder':
        r = st.slider('Cylinder Radius r (cm)', 1.0, 5.0, 3.0, 0.01)
        h, A = cylinder_given_r(r)
        st.markdown(f"""
**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A:** {A:.2f} cm²
""")
    elif shape == 'Frustum':
        rt = st.slider('Top Radius rt (cm)', 1.0, 5.0, 2.8, 0.01)
        rb = st.slider('Bottom Radius rb (cm)', 1.0, 5.0, 3.2, 0.01)
        h, A = frustum_given_rt_rb(rt, rb)
        st.markdown(f"""
**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A:** {A:.2f} cm²
""")
    elif shape == 'Cuboid':
        L = st.slider('Cuboid Length L (cm)', 1.0, 10.0, 6.0, 0.01)
        W = st.slider('Cuboid Width W (cm)', 1.0, 10.0, 6.0, 0.01)
        H, A = cuboid_given_LW(L, W)
        st.markdown(f"""
**Height H (for 330 cm³):** {H:.2f} cm

**Surface Area A:** {A:.2f} cm²
""")
    elif shape == 'Cube':
        a, A = cube_fixed_volume()
        st.markdown(f"""
**Edge a (fixed by 330 cm³):** {a:.2f} cm

**Surface Area A:** {A:.2f} cm²
""")
    else:
        b = st.slider('Base Side b (cm)', 3.0, 10.0, 6.0, 0.01)
        h, A = square_pyramid_given_b(b)
        st.markdown(f"""
**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A (incl. base):** {A:.2f} cm²
""")

with col2:
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_zlabel('z (cm)')

    if shape == 'Cylinder' and r is not None and h is not None:
        theta = np.linspace(0, 2*np.pi, 80)
        z = np.linspace(0, h, 60)
        Theta, Z = np.meshgrid(theta, z)
        X = r * np.cos(Theta)
        Y = r * np.sin(Theta)
        ax.plot_surface(X, Y, Z, color='#3b82f6', alpha=0.7, linewidth=0)
        rr = np.linspace(0, r, 40)
        th = np.linspace(0, 2*np.pi, 80)
        RR, TH = np.meshgrid(rr, th)
        Xc = RR * np.cos(TH)
        Yc = RR * np.sin(TH)
        ax.plot_surface(Xc, Yc, np.zeros_like(Xc), color='#eab308', alpha=0.5, linewidth=0)
        ax.plot_surface(Xc, Yc, h*np.ones_like(Xc), color='#eab308', alpha=0.5, linewidth=0)
    elif shape == 'Frustum' and rt is not None and rb is not None and h is not None:
        theta = np.linspace(0, 2*np.pi, 80)
        z = np.linspace(0, h, 60)
        Theta, Z = np.meshgrid(theta, z)
        Rz = rb + (rt - rb) * (Z / h)
        X = Rz * np.cos(Theta)
        Y = Rz * np.sin(Theta)
        ax.plot_surface(X, Y, Z, color='#3b82f6', alpha=0.7, linewidth=0)
        rr_b = np.linspace(0, rb, 40)
        rr_t = np.linspace(0, rt, 40)
        th = np.linspace(0, 2*np.pi, 80)
        RRb, TH = np.meshgrid(rr_b, th)
        RRt, THt = np.meshgrid(rr_t, th)
        Xb = RRb * np.cos(TH)
        Yb = RRb * np.sin(TH)
        Xt = RRt * np.cos(THt)
        Yt = RRt * np.sin(THt)
        ax.plot_surface(Xb, Yb, np.zeros_like(Xb), color='#eab308', alpha=0.5, linewidth=0)
        ax.plot_surface(Xt, Yt, h*np.ones_like(Xt), color='#eab308', alpha=0.5, linewidth=0)
    elif shape == 'Cuboid' and L is not None and W is not None and H is not None:
        x0, x1 = -W/2.0, W/2.0
        y0, y1 = 0.0, L
        z0, z1 = 0.0, H
        # draw 12 edges for simplicity
        edges = [
            ([x0,x1],[y0,y0],[z0,z0]), ([x1,x1],[y0,y1],[z0,z0]),
            ([x1,x0],[y1,y1],[z0,z0]), ([x0,x0],[y1,y0],[z0,z0]),
            ([x0,x1],[y0,y0],[z1,z1]), ([x1,x1],[y0,y1],[z1,z1]),
            ([x1,x0],[y1,y1],[z1,z1]), ([x0,x0],[y1,y0],[z1,z1]),
            ([x0,x0],[y0,y0],[z0,z1]), ([x1,x1],[y0,y0],[z0,z1]),
            ([x1,x1],[y1,y1],[z0,z1]), ([x0,x0],[y1,y1],[z0,z1]),
        ]
        for ex, ey, ez in edges:
            ax.plot(ex, ey, ez, color='#3b82f6')
    elif shape == 'Cube' and a is not None:
        x0, x1 = -a/2.0, a/2.0
        y0, y1 = 0.0, a
        z0, z1 = 0.0, a
        edges = [
            ([x0,x1],[y0,y0],[z0,z0]), ([x1,x1],[y0,y1],[z0,z0]),
            ([x1,x0],[y1,y1],[z0,z0]), ([x0,x0],[y1,y0],[z0,z0]),
            ([x0,x1],[y0,y0],[z1,z1]), ([x1,x1],[y0,y1],[z1,z1]),
            ([x1,x0],[y1,y1],[z1,z1]), ([x0,x0],[y1,y0],[z1,z1]),
            ([x0,x0],[y0,y0],[z0,z1]), ([x1,x1],[y0,y0],[z0,z1]),
            ([x1,x1],[y1,y1],[z0,z1]), ([x0,x0],[y1,y1],[z0,z1]),
        ]
        for ex, ey, ez in edges:
            ax.plot(ex, ey, ez, color='#3b82f6')
    elif shape == 'Square Pyramid' and b is not None and h is not None:
        half_b = b/2.0
        base = np.array([[-half_b,-half_b,0], [half_b,-half_b,0], [half_b,half_b,0], [-half_b,half_b,0]])
        apex = np.array([0,0,h])
        # base
        for i in range(4):
            j = (i+1) % 4
            ax.plot([base[i,0], base[j,0]], [base[i,1], base[j,1]], [0,0], color='#eab308')
        # sides
        for i in range(4):
            ax.plot([base[i,0], apex[0]], [base[i,1], apex[1]], [0, apex[2]], color='#3b82f6')

    ax.view_init(elev=20, azim=35)
    st.subheader('3D View (拖動旋轉需以滑鼠；觸控可兩指縮放)')
    st.pyplot(fig, use_container_width=True)

st.info('若你的環境無法安裝 Plotly，本版本改用 Matplotlib 以避免 ModuleNotFoundError。')
