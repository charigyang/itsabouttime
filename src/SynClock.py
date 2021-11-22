import numpy as np
import cv2
import math
import random

def truefalse(p):
	return random.random() < p

def intminmax(mi, ma):
	return random.choice(range(mi, ma+1))

def minmax(mi, ma):
	return np.random.uniform(mi,ma)

def rand_colour(p_gray=0, p_light=0, p_dark=0, p_red=0):
	xmin = 0; xmax = 255;
	gray = random.random() < p_gray
	if gray:
		x = random.choice(range(xmin,xmax))
		return (x, x, x)
	light = random.random() < p_light
	if light: 
		xmin  = 200;
		x = random.choice(range(xmin,xmax))
		return (x, x, x)
	dark = random.random() < p_dark
	if dark: 
		xmax = 100;
		x = random.choice(range(xmin,xmax))
		return (x, x, x)
	
	xmin = 50; ymin = 50; zmin = 50;
	xmax = 200; ymax = 200; zmax = 200;
	red = random.random() < p_red
	if red:
		xmin = 0; ymin = 0; zmin = 127;
		xmax = 100; ymax = 100; zmax = 255;
	x = random.choice(range(xmin,xmax))
	y = random.choice(range(ymin,ymax))
	z = random.choice(range(zmin,zmax))
	return (x, y, z)

def get_coordinates(cx, cy, r, scale, back_scale, a):
	x1 = (cx + scale*r * np.cos(a * math.pi/180)).astype(int)
	y1 = (cy + scale*r * np.sin(a * math.pi/180)).astype(int)
	x2 = (cx + back_scale*r * np.cos(a * math.pi/180)).astype(int)
	y2 = (cy + back_scale*r * np.sin(a * math.pi/180)).astype(int)
	return (x1, y1), (x2, y2)

def draw_line(img, source, dest, colour, thickness, arrow=False, arrow_scale=None, tip_length=None, shadow=False, rand=False):
	img = cv2.line(img, source, dest, colour, thickness)
	x1, y1 = source
	x2, y2 = dest

	if arrow:
		arrowhead = (int(x2 + (x1-x2) * arrow_scale), int(y2 + (y1-y2) * arrow_scale))
		img = cv2.arrowedLine(img, dest, arrowhead, colour, thickness, tipLength=tip_length)

	if shadow:
		dx = random.choice([1,-1]) * intminmax(thickness, 30)
		dy = random.choice([1,-1]) * intminmax(thickness, 30)
		shadow_colour = colour if truefalse(0.5) else rand_colour(p_dark=0.8)
		shadow_alpha = minmax(0.1,0.9)
		img_orig = img.copy()
		img_shadow = cv2.line(img, (x1+dx, y1+dy), (x2+dx, y2+dy), shadow_colour, thickness)
		img = cv2.addWeighted(img_shadow, shadow_alpha, img_orig, 1-shadow_alpha, 0)		
	return img

def draw_random_lines(img, cx, cy, r, R, num=3):
	for _ in range(num):
		r1 = intminmax(r,R)
		r2 = intminmax(r,R)
		theta1 = minmax(0, 360)
		theta2 = minmax(0, 360)
		colour = rand_colour()
		thickness = intminmax(1,10)

		x1 = (cx + r1 * np.cos(theta1 * math.pi/180)).astype(int)
		y1 = (cy + r1 * np.sin(theta1 * math.pi/180)).astype(int)
		x2 = (cx + r2 * np.cos(theta2 * math.pi/180)).astype(int)
		y2 = (cy + r2 * np.sin(theta2 * math.pi/180)).astype(int)

		img_shadow = cv2.line(img, (x1, y1), (x2, y2), colour, thickness)
	return img

def gen_clock(use_homography=True, use_artefacts=True):
	#hyperparameters:
	#canvas
	H = 448
	W = 448
	h = 392 #intminmax(384, 512)
	w = h 
	use_border = True
	canvas_background_colour = rand_colour(p_gray=0.2)
	hmax = 0.1

	#clock shape
	use_rectangle_clock = truefalse(0.25)
	clock_center_coordinates = (h//2, w//2)
	clock_border_thickness = intminmax(0, 60)
	clock_radius = (min(h,w)//2 - clock_border_thickness//2 -1)
	clock_background_colour = rand_colour(p_light=0.8) if truefalse(0.7) else rand_colour(p_gray=0.2)
	clock_border_colour = rand_colour(p_gray=0)

	#ticks (minute)
	use_rectangle_tick = truefalse(0.6) if use_rectangle_clock else False 
	use_minute_tick = truefalse(0.8)
	tick_gap = intminmax(0, 15)
	tick_length = intminmax(1, 10)
	tick_thickness = intminmax(1, 10)
	tick_colour = rand_colour(p_dark=0.5, p_gray=0.2)

	#ticks (hour)
	tick_h_gap = tick_gap
	tick_h_length = intminmax(tick_length, 15)
	tick_h_thickness = intminmax(tick_thickness, 15)
	tick_h_colour = tick_colour if truefalse(0.8) else rand_colour(p_dark=0.5, p_gray=0.2)

	#numerals
	use_numerals = truefalse(0.8)
	use_roman = truefalse(0.3)
	num_rotate = truefalse(0.3) if use_roman else truefalse(0.05)
	num_font = intminmax(0,7)
	num_font_scale = minmax(0.5,2)
	num_font_thickness = intminmax(1,4)
	num_colour = tick_colour if truefalse(0.8) else rand_colour(p_dark=0.5, p_gray=0.2)
	num_gap = intminmax(10, 40)

	#hands
	use_alarm_hand = truefalse(0.4)
	time_alarm = minmax(0, 1)
	alarm_scale = minmax(0.2, 0.5)
	alarm_back_scale = minmax(-0.1,0)
	alarm_colour = rand_colour(p_light=0.25, p_gray=0.25, p_dark=0.25)
	alarm_thickness = intminmax(1,5)

	use_second_hand = truefalse(0.5)
	time_hour = intminmax(0, 12)
	time_minute = intminmax(0, 60)
	time_second = intminmax(0, 60)
	sec_scale = minmax(0.8, 0.95)
	sec_back_scale = 0 if truefalse(0.3) else minmax(-0.3, 0)
	sec_colour = rand_colour(p_red=0.4, p_gray=0.15, p_dark=0.2)
	sec_thickness = intminmax(1,5)

	min_arrowed = truefalse(0.5)
	min_arrow_scale = minmax(0.3, 1)
	min_tip_length = minmax(0.1, 0.2)
	min_scale = minmax(0.6, sec_scale)
	min_back_scale = 0 if truefalse(0.3) else minmax(-0.3, 0)
	min_colour = rand_colour(p_dark=0.9)
	min_thickness = intminmax(5,12)

	hr_arrowed = min_arrowed
	hr_arrow_scale = min_arrow_scale
	hr_scale = minmax(0.3, min_scale)
	hr_tip_length = min_tip_length * min_scale / hr_scale
	hr_back_scale = 0 if truefalse(0.3) else minmax(-0.15, 0)
	hr_colour = min_colour if truefalse(0.8) else rand_colour(p_dark=0.9)
	hr_thickness = intminmax(min_thickness,18)

	#circle
	use_circle_border = truefalse(0.5)
	circle_radius = intminmax(8,12)
	circle_colour = rand_colour(p_dark=0.5)
	circle_border_colour = rand_colour(p_dark=0.1)
	circle_border_thickness = intminmax(1,3)

	#create background
	img = np.zeros((h, w, 3), np.uint8)
	img[:] = canvas_background_colour

	#shadow
	if use_artefacts:
		hr_shadow = truefalse(0.5)
		min_shadow = truefalse(0.5)
		sec_shadow = truefalse(0.5)
		alarm_shadow = truefalse(0.5)

		#random
		num_random_lines = intminmax(0, 5)
	else:
		hr_shadow = False
		min_shadow = False
		sec_shadow = False
		alarm_shadow = False

	#create clock
	if not use_rectangle_clock:
		img = cv2.circle(img, clock_center_coordinates, clock_radius, clock_background_colour, cv2.FILLED)
		img = cv2.circle(img, clock_center_coordinates, clock_radius, clock_border_colour, clock_border_thickness)
	else:
		img = cv2.rectangle(img, (0, 0), (h, w), clock_background_colour, cv2.FILLED)
		img = cv2.rectangle(img, (0, 0), (h, w), clock_border_colour, clock_border_thickness)

	#create ticks
	cy, cx = clock_center_coordinates
	r = clock_radius
	a = np.arange(60)*6
	acos = np.cos(a * math.pi/180)
	asin = np.sin(a * math.pi/180)
	if use_rectangle_tick:
		atan = np.tan(a * math.pi/180)
		acos[0:8] = 1 ; acos[23:38] = -1 ; acos[53:] = 1
		asin[0:8] = atan[0:8] ; asin[23:38] = atan[23:38] ; asin[53:] = atan[53:]
		asin[8:23] = -1 ; asin[38:53] = 1 ;
		acos[8:23] = 1/atan[8:23] ; acos[38:53] = 1/atan[38:53];
		
	x1 = np.rint(cx + (r-clock_border_thickness-tick_gap) * acos).astype(int)
	y1 = np.rint(cy + (r-clock_border_thickness-tick_gap) * asin).astype(int)
	x2 = np.rint(cx + (r-clock_border_thickness-tick_gap-tick_length) * acos).astype(int)
	y2 = np.rint(cy + (r-clock_border_thickness-tick_gap-tick_length) * asin).astype(int)
	h_x1 = np.rint(cx + (r-clock_border_thickness-tick_h_gap) * acos).astype(int)
	h_y1 = np.rint(cy + (r-clock_border_thickness-tick_h_gap) * asin).astype(int)
	h_x2 = np.rint(cx + (r-clock_border_thickness-tick_h_gap-tick_h_length) * acos).astype(int)
	h_y2 = np.rint(cy + (r-clock_border_thickness-tick_h_gap-tick_h_length) * asin).astype(int)
	for i in range(len(a)):
		if i % 5 == 0:
			img = cv2.line(img, (h_x1[i], h_y1[i]), (h_x2[i], h_y2[i]), tick_h_colour, tick_h_thickness)	
		elif use_minute_tick:
			img = cv2.line(img, (x1[i], y1[i]), (x2[i], y2[i]), tick_colour, tick_thickness)

	#create numerals
	if use_numerals:
		num_texts = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
		if use_roman: num_texts = ['III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'I', 'II', 'III']
		a2 = np.arange(12) * 30
		acos2 = np.cos(a2 * math.pi/180)
		asin2 = np.sin(a2 * math.pi/180)
		if use_rectangle_tick:
			atan2 = np.tan(a2 * math.pi/180)
			acos2[[0, 1, 11]] = 1 ; acos2[[5, 6, 7]] = -1 ;
			asin2[[0, 1, 11]] = atan2[[0, 1, 11]] ; asin2[[5,6,7]] = -atan2[[5,6,7]] ;
			asin2[[2,3,4]] = 1 ; asin2[[8,9,10]] = -1 ;
			acos2[[2,3,4]] = 1/atan2[[2,3,4]] ; acos2[[8,9,10]] = -1/atan2[[8,9,10]];
		tx = np.rint(cx + (r-clock_border_thickness-tick_h_gap-tick_h_length-num_gap) * acos2).astype(int)
		ty = np.rint(cy + (r-clock_border_thickness-tick_h_gap-tick_h_length-num_gap) * asin2).astype(int)
		for i in range(12):
			textsize = cv2.getTextSize(str(num_texts[i]), num_font, num_font_scale, num_font_thickness)[0]
			textX = tx[i] - textsize[0]//2
			textY = ty[i] + textsize[1]//2
			cv2.putText(img, str(num_texts[i]), (textX, textY), num_font, num_font_scale, num_colour, num_font_thickness)

	#hands
	alarm = time_alarm
	a_alarm = alarm * 360
	if use_alarm_hand:
		source, dest = get_coordinates(cx, cy, r, alarm_scale, alarm_back_scale, a_alarm)
		img = draw_line(img, source, dest, alarm_colour, alarm_thickness, shadow=alarm_shadow)

	second = time_second
	a_second = second * 6 - 90
	if use_second_hand:
		source, dest = get_coordinates(cx, cy, r, sec_scale, sec_back_scale, a_second)
		img = draw_line(img, source, dest, sec_colour, sec_thickness, shadow=sec_shadow)

	minute = time_minute + second/60
	a_minute = minute * 6 - 90
	source, dest = get_coordinates(cx, cy, r, min_scale, min_back_scale, a_minute)
	img = draw_line(img, source, dest, min_colour, min_thickness, min_arrowed, min_arrow_scale, min_tip_length, shadow=min_shadow)

	hour = time_hour + minute/60
	a_hour = hour * 30 - 90
	source, dest = get_coordinates(cx, cy, r, hr_scale, hr_back_scale, a_hour)
	img = draw_line(img, source, dest, hr_colour, hr_thickness, hr_arrowed, hr_arrow_scale, hr_tip_length, shadow=hr_shadow)
	
	if use_artefacts:
		img = draw_random_lines(img, cx, cy, circle_radius, r, num=num_random_lines)

	#circle
	img = cv2.circle(img, clock_center_coordinates, circle_radius, circle_colour, cv2.FILLED)
	if use_circle_border:
		img = cv2.circle(img, clock_center_coordinates, circle_radius, circle_border_colour, circle_border_thickness)

	if use_border:
		IMG = np.zeros((H, W, 3), np.uint8)
		IMG[:] = canvas_background_colour
		Iy = (H-h)//2
		Ix = (W-w)//2
		IMG[Iy:Iy+h, Ix:Ix+w, :] = img
		img = IMG

	if use_homography:
		points = np.array(((Ix,Iy), (Ix+w,Iy), (Ix,Iy+h), (Ix+w,Iy+h)), dtype=np.float32)
		f = 4
		purturb = np.random.randint(-Ix*f,Ix*f+1,(4,2)).astype(np.float32)
		points2 = points + purturb
		M = cv2.getPerspectiveTransform(points, points2)
		img = cv2.warpPerspective(img, M, (H, W), borderValue=canvas_background_colour)
		Minv = cv2.findHomography(points2*2/448-1, points*2/448-1)[0]
		
	else:
		Minv = np.array([[1.,0,0],[0,1.,0],[0,0,1.]]).astype(np.float32)

	return img, time_hour, time_minute, Minv
