function y = f_squeak_generator(squeak_type, Fs, duration, freq_low, freq_high)
% 1 line sweep up
% 2 line sweep down
% 3 chirp up
% 4 chirp down
% 5 reverse chirp up
% 6 reverse chirp down
% 7 parabola up
% 8 parabola down

freq_low = freq_low*1000;
freq_high = freq_high * 1000;

if squeak_type == 0
    t = 0:1/Fs:duration;
    y = sin(t*2*pi*freq_low);
elseif squeak_type == 1
    t = 0:1/Fs:duration;
    y = chirp(t,freq_low,duration,freq_high);
elseif squeak_type == 2
    t = 0:1/Fs:duration;
    y = chirp(t,freq_high,duration,freq_low);
elseif squeak_type == 3
    t = 0:1/Fs:duration;
    y = chirp(t,freq_low,duration,freq_high,'quadratic',[],'concave');
elseif squeak_type == 4
    t = 0:1/Fs:duration;
    y = chirp(t,freq_high,duration,freq_low,'quadratic',[],'convex');
elseif squeak_type == 5
    t = -duration:1/Fs:0;
    y = chirp(t,freq_low,duration,freq_high,'quadratic',[],'concave');
elseif squeak_type == 6
    t =  0:1/Fs:duration;
    y = chirp(t,freq_low,duration,freq_high,'quadratic',[],'convex');
elseif squeak_type == 7
    t = -duration/2:1/Fs:duration/2;
    y = chirp(t,freq_low,duration/2,freq_high,'quadratic',[],'concave');
elseif squeak_type == 8
    t = -duration/2:1/Fs:duration/2;
    y = chirp(t,freq_low,duration/2,freq_high,'quadratic',[],'convex');
end




end
