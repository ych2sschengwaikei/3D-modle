
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

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

# --- UI ---
st.set_page_config(page_title='330 mL Container Surface Area (Web)', layout='wide')
st.title('330 mL Container Surface Area Calculator')

shape = st.radio('選擇形狀 Shape', ['Cylinder', 'Frustum', 'Cuboid', 'Cube', 'Square Pyramid'], horizontal=True)

col1, col2 = st.columns([1,1])

with col1:
    if shape == 'Cylinder':
        r = st.slider('Cylinder Radius r (cm)', 1.0, 5.0, 3.0, 0.01)
        h, A = cylinder_given_r(r)
        st.markdown(f"**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A:** {A:.2f} cm²")
    elif shape == 'Frustum':
        rt = st.slider('Top Radius rt (cm)', 1.0, 5.0, 2.8, 0.01)
        rb = st.slider('Bottom Radius rb (cm)', 1.0, 5.0, 3.2, 0.01)
        h, A = frustum_given_rt_rb(rt, rb)
        st.markdown(f"**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A:** {A:.2f} cm²")
    elif shape == 'Cuboid':
        L = st.slider('Cuboid Length L (cm)', 1.0, 10.0, 6.0, 0.01)
        W = st.slider('Cuboid Width W (cm)', 1.0, 10.0, 6.0, 0.01)
        H, A = cuboid_given_LW(L, W)
        st.markdown(f"**Height H (for 330 cm³):** {H:.2f} cm

**Surface Area A:** {A:.2f} cm²")
    elif shape == 'Cube':
        a, A = cube_fixed_volume()
        st.markdown(f"**Edge a (fixed by 330 cm³):** {a:.2f} cm

**Surface Area A:** {A:.2f} cm²")
    else:
        b = st.slider('Base Side b (cm)', 3.0, 10.0, 6.0, 0.01)
        h, A = square_pyramid_given_b(b)
        st.markdown(f"**Height h (for 330 cm³):** {h:.2f} cm

**Surface Area A (incl. base):** {A:.2f} cm²")

with col2:
    # 3D Plotly figure
    face_color = 'rgba(59,130,246,0.6)'
    cap_color = 'rgba(234,179,8,0.6)'
    fig = go.Figure()

    if shape == 'Cylinder':
        r = r
        theta = np.linspace(0, 2*np.pi, 80)
        z = np.linspace(0, h, 60)
        Theta, Z = np.meshgrid(theta, z)
        X = r * np.cos(Theta)
        Y = r * np.sin(Theta)
        fig.add_surface(x=X, y=Y, z=Z, showscale=False, opacity=0.9)
        rr = np.linspace(0, r, 40)
        th = np.linspace(0, 2*np.pi, 80)
        RR, TH = np.meshgrid(rr, th)
        Xc = RR * np.cos(TH)
        Yc = RR * np.sin(TH)
        fig.add_surface(x=Xc, y=Yc, z=np.zeros_like(Xc), showscale=False, opacity=0.6)
        fig.add_surface(x=Xc, y=Yc, z=h*np.ones_like(Xc), showscale=False, opacity=0.6)
    elif shape == 'Frustum':
        theta = np.linspace(0, 2*np.pi, 80)
        z = np.linspace(0, h, 60)
        Theta, Z = np.meshgrid(theta, z)
        Rz = rb + (rt - rb) * (Z / h)
        X = Rz * np.cos(Theta)
        Y = Rz * np.sin(Theta)
        fig.add_surface(x=X, y=Y, z=Z, showscale=False, opacity=0.9)
        rr_b = np.linspace(0, rb, 40)
        rr_t = np.linspace(0, rt, 40)
        th = np.linspace(0, 2*np.pi, 80)
        RRb, TH = np.meshgrid(rr_b, th)
        RRt, THt = np.meshgrid(rr_t, th)
        Xb = RRb * np.cos(TH)
        Yb = RRb * np.sin(TH)
        Xt = RRt * np.cos(THt)
        Yt = RRt * np.sin(THt)
        fig.add_surface(x=Xb, y=Yb, z=np.zeros_like(Xb), showscale=False, opacity=0.6)
        fig.add_surface(x=Xt, y=Yt, z=h*np.ones_like(Xt), showscale=False, opacity=0.6)
    elif shape == 'Cuboid':
        # Draw a box using Mesh3d
        x0, x1 = -W/2.0, W/2.0
        y0, y1 = 0.0, L
        z0, z1 = 0.0, H
        X = [x0,x1,x1,x0, x0,x1,x1,x0]
        Y = [y0,y0,y1,y1, y0,y0,y1,y1]
        Z = [z0,z0,z0,z0, z1,z1,z1,z1]
        I = [0,1,2,3, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7]
        fig.add_mesh3d(x=X, y=Y, z=Z, i=[0,1,2,3, 4,5], j=[1,2,3,0,5], k=[2,3,0,1,6], opacity=0.5)
        # Note: Mesh3d faces simplified; visual is adequate for web demo
    elif shape == 'Cube':
        a, _ = cube_fixed_volume()
        x0, x1 = -a/2.0, a/2.0
        y0, y1 = 0.0, a
        z0, z1 = 0.0, a
        X = [x0,x1,x1,x0, x0,x1,x1,x0]
        Y = [y0,y0,y1,y1, y0,y0,y1,y1]
        Z = [z0,z0,z0,z0, z1,z1,z1,z1]
        fig.add_mesh3d(x=X, y=Y, z=Z, opacity=0.5)
    else:
        half_b = b/2.0
        base_x = np.array([-half_b, half_b, half_b, -half_b])
        base_y = np.array([-half_b, -half_b, half_b, half_b])
        base_z = np.zeros(4)
        apex = np.array([0.0, 0.0, h])
        # Base
        fig.add_mesh3d(x=base_x, y=base_y, z=base_z, opacity=0.6)
        # Sides (triangles)
        for i in range(4):
            j = (i+1) % 4
            x = [base_x[i], base_x[j], apex[0]]
            y = [base_y[i], base_y[j], apex[1]]
            z = [base_z[i], base_z[j], apex[2]]
            fig.add_mesh3d(x=x, y=y, z=z, opacity=0.6)

    fig.update_traces(colorscale=[[0, '#3b82f6'], [1, '#3b82f6']])
    fig.update_layout(scene=dict(xaxis_title='x (cm)', yaxis_title='y (cm)', zaxis_title='z (cm)'),
                      height=600, margin=dict(l=0, r=0, t=0, b=0))
    st.subheader('3D View (拖動旋轉 Rotate by dragging)')
    st.plotly_chart(fig, use_container_width=True)

st.info('點選上方選項及滑桿，即可即時看到體積固定為 330 mL 的不同形狀及其表面積。')
