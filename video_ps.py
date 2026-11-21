# PS only implementation of contrast_enhance.py
#
# Author: Jindrich Vodak <xvodak06@stud.fit.vut.cz>
# Date: 20. 11. 2026

import argparse
import cv2
import numpy
import time
import threading
import math
import cProfile
import pstats



# ##############################################################################
# Debug function to dump frame in a text format
# ##############################################################################
def debug_dump(frame, file):
    rows, cols, c3 = frame.shape
    for r in range(rows):
        for c in range(cols):
            B, G, R = frame[r,c]
            file.write(bytes([B, G, R]))


# ##############################################################################
# Color conversions
# ##############################################################################

# Conversion matrixes (JPEG standard)
# Sources:
#    https://docs.opencv.org/3.4.15/de/d25/imgproc_color_conversions.html
#    https://en.wikipedia.org/wiki/YCbCr
MATRIX_BGR2YCrCb = numpy.array([ 
    [0.114, -0.081312,  0.5],   
    [0.587, -0.418688, -0.331264],
    [0.299,  0.5,      -0.168736],
])
MATRIX_YCrCb2BGR = numpy.array([ 
    [1,      1,        1],   
    [0,     -0.714136, 1.402],
    [1.772, -0.344136, 0],
])

# Color convertion from BGR representation to YCrCb representation
# Note: ignoring delta of 128 so Cr and Cb are in <-128,127> range
def frame_convert_BGR2YCrCb(frame):
    return numpy.matmul(frame, MATRIX_BGR2YCrCb)
def pixel_convert_BGR2YCrCb(B, G, R):
    Y  = MATRIX_BGR2YCrCb[0,0]*B + MATRIX_BGR2YCrCb[1,0]*G + MATRIX_BGR2YCrCb[2,0]*R
    Cr = MATRIX_BGR2YCrCb[0,1]*B + MATRIX_BGR2YCrCb[1,1]*G + MATRIX_BGR2YCrCb[2,1]*R
    Cb = MATRIX_BGR2YCrCb[0,2]*B + MATRIX_BGR2YCrCb[1,2]*G + MATRIX_BGR2YCrCb[2,2]*R
    return Y, Cr, Cb

# Color convertion from YCrCb representation to BGR representation
# Note: ignoring delta of 128 again to be compatible with the previous function (expects Cr and Cb from <-128,127> range)
def frame_convert_YCrCb2BGR(frame):
    frame = numpy.matmul(frame, MATRIX_YCrCb2BGR) # base conversion
    return frame
def pixel_convert_YCrCb2BGR(Y, Cr, Cb):
    B = MATRIX_YCrCb2BGR[0,0]*Y + MATRIX_YCrCb2BGR[1,0]*Cr + MATRIX_YCrCb2BGR[2,0]*Cb
    G = MATRIX_YCrCb2BGR[0,1]*Y + MATRIX_YCrCb2BGR[1,1]*Cr + MATRIX_YCrCb2BGR[2,1]*Cb
    R = MATRIX_YCrCb2BGR[0,2]*Y + MATRIX_YCrCb2BGR[1,2]*Cr + MATRIX_YCrCb2BGR[2,2]*Cb
    return B, G, R

# Type conversion (rounding) of color data from float to unsigned byte
def frame_float2byte(frame):
    frame = numpy.where(frame<0.0, 0.0, frame) # threshold over/under flows of <0,255> range in BGR values
    frame = numpy.where(frame>255.0, 255.0, frame)
    #return numpy.around(frame).astype(numpy.uint8) # around is painfully slow!
    return (frame+0.5).astype(numpy.uint8) # faster hack with 0.5 increment converting astype number truncating (flooring) to rounding
def pixel_float2byte(B, G, R):
    B = min(max(B, 0.0), 255.0) # threshold over/under flow of <0,255> range in BGR values
    G = min(max(G, 0.0), 255.0)
    R = min(max(R, 0.0), 255.0)
    return numpy.array([round(B), round(G), round(R)], dtype=numpy.uint8)



# ##############################################################################
# Histogram equalization
# ##############################################################################
# Implementation of WTHE method with r=0.5 and v=0.5 for Y values range <0,255>
# Source: https://ieeexplore.ieee.org/document/4266969
class WTHE():
    # Create empty data structures
    def __init__(self):
        self.histogram = numpy.zeros(256) # histogram of input values
        self.transform = numpy.zeros(256) # transformation lookup table
        self.transform_ready = False      # transformation ready (table is filled) flag
        self.sum_before = 0               # total sum of values before transformation (used for debug)
        self.sum_after = 0                # total sum of values after transformation (used for debug)
        self.values = 0                   # number of processed values (used for debug)

    # Zero-out (restart) the histogram accumulation 
    def restart(self):
        self.histogram = numpy.zeros(256)
        self.sum_before = 0
        self.sum_after = 0
        self.values = 0

    # Add input value to histogram and perform value transformation
    def process_value(self, value):
        rval = int(round(value))
        self.histogram[rval] += 1 # histogram update
        self.values += 1
        self.sum_before += rval
        if self.transform_ready: # transformation
            rval = self.transform[rval]
        self.sum_after += rval
        return rval

    # Add array of input values to histogram and perform their transformation    
    def process_values(self, values):
        values = (values+0.5).astype(numpy.uint8) # hack for fast rounding in type conversion over array
        self.histogram += numpy.histogram(values, 256, (0,256))[0] # compute and accumulate histogram of the whole array
        self.values += values.size
        self.sum_before += values.sum()
        if self.transform_ready: # apply transformation LUT onto each element of the array
            values = self.transform[values]
        self.sum_after += values.sum()
        return values

    # Compute new transformation table based on histogram values using WTHE method
    def transform_update(self, id):
        self.transform_ready = False
        P = self.histogram / self.values # convert histogram to probability of individual items
        Pl = 0.0003      # lower threshold is a really small percentage
        Pu = P.max() / 2 # upper threshold is half (v=0.5) of the highest probability
        Pr = Pu - Pl     # size of probability range between upper and lower bound
        cumsum = 0.0
        table = numpy.zeros(256) 
        for k in range(256): # cumulative sum over P with thresholds and weigth application 
            if P[k] < Pl: # thresholding of Pk
                Pwt = 0.0
            elif P[k] > Pu:
                Pwt = Pu
            else: # weighted probability transformation with r = 0.5 (sqrt)
                Pwt = math.sqrt((P[k] - Pl) / Pr) * Pu
            cumsum += Pwt
            table[k] = cumsum * 255.0 # transformation table converting k to (L-1)*Ck (note that cumsum == Ck here)
        table /= cumsum # normalization as total Pwt sum can be bigger than 1 (note that sum(Pwt) == cumsum here)
        self.transform = numpy.around(table).astype(numpy.uint8) # round and store transformation table
        self.transform_ready = True                              # set the table as valid
        if self.values:
            print(
                "Frame:", id,
                "\tMeans:", "{:.3f}".format(self.sum_before/self.values), "{:.3f}".format(self.sum_after/self.values),
                "\tCumSum:", "{:.3f}".format(cumsum), "\tPu:", "{:.3f}".format(Pu*100.0)
            )
        self.restart() # clear histogram


# ##############################################################################
# Frame processing
# ##############################################################################

# Python native iteration over frame pixels
def processing_native(frame, Yhe):
    rows, cols, c3 = frame.shape
    for r in range(rows): # looping over pixels in frame
        for c in range(cols):
            B, G, R = frame[r,c]                         # pixel color channels decoding
            Y, Cr, Cb = pixel_convert_BGR2YCrCb(B, G, R) # convert to YCrCb
            Y = Yhe.process_value(Y)                     # update Y histogram and transform/equalize Y values
            B, G, R = pixel_convert_YCrCb2BGR(Y, Cr, Cb) # convert back to BGR
            frame[r,c] = pixel_float2byte(B, G, R)       # round to 8-bit color values
    return frame

# Numpy accelerated iteration over frame pixels
def processing_numpy(frame, Yhe):
    frame = frame_convert_BGR2YCrCb(frame)          # convert to YCrCb
    frame[:,:,0] = Yhe.process_values(frame[:,:,0]) # update Y histogram and transform/equalize Y values
    frame = frame_convert_YCrCb2BGR(frame)          # convert back to BGR
    frame = frame_float2byte(frame)                 # round to 8-bit color values
    return frame



# ##############################################################################
# Main program body
# ##############################################################################
def main():
    # Preserve the original command-line interface; PYNQ HDMI replaces the
    # default camera and graphical-window paths.
    parser = argparse.ArgumentParser(description='Adaptive contrast enhancement using WTHE method on PYNQ-Z2.')
    parser.add_argument('-i', metavar='PATH', dest='input', default=None,
                        help='input video file (default: HDMI input)')
    parser.add_argument('-o', metavar='PATH', dest='output', default=None,
                        help='output MP4 file (default: HDMI output)')
    parser.add_argument('-l', metavar='N', type=int, dest='loops', default=1,
                        help='loop input file N times (default: 1, 0 means forever)')
    parser.add_argument('-c', metavar='N', type=int, dest='max_frames', default=0,
                        help='stop after N frames (default: 0, no limit)')
    parser.add_argument('-n', dest='native', action='store_true',
                        help='use slow native Python pixel processing')
    parser.add_argument('-S', metavar='PATH', dest='simulation', default=None,
                        help='dump raw input/output frames to PATH.in and PATH.out')
    parser.add_argument('-p', dest='profile', action='store_true',
                        help='enable cProfile; print statistics and save video_ps.prof')
    args = parser.parse_args()
    if args.loops < 0:
        args.loops = 0
    if args.max_frames < 1:
        args.max_frames = -1

    input_capture = None
    hdmi_in = None
    hdmi_out = None
    output_writer = None
    sim_in = None
    sim_out = None
    HE_thread = None
    profiler = cProfile.Profile() if args.profile else None
    frames = 0
    start = None
    end = None
    error = None

    try:
        print('Preparing input and output ...')
        # A PYNQ base overlay is needed whenever either side uses HDMI.
        # Import it lazily so file-to-file runs also work off-board.
        if args.input is None or args.output is None:
            from pynq import Overlay
            from pynq.lib.video import VideoMode, PIXEL_BGR
            base = BaseOverlay('base.bit')

        if args.input is None:
            hdmi_in = base.video.hdmi_in
            hdmi_in.configure(PIXEL_BGR)
            hdmi_in.start()
            mode = hdmi_in.mode
            width, height = mode.width, mode.height
            fps = mode.fps
            print('HDMI input: {}x{} at {} FPS'.format(width, height, fps))
        else:
            input_capture = cv2.VideoCapture(args.input)
            if not input_capture.isOpened():
                raise RuntimeError('ERROR: Unable to read input file data!')
            fps = input_capture.get(cv2.CAP_PROP_FPS)
            width = int(input_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(input_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if args.output is None:
            hdmi_out = base.video.hdmi_out
            hdmi_out.configure(VideoMode(1280, 720, 24, 60), PIXEL_BGR)
            hdmi_out.start()
        else:
            # A container/codec can be absent on the PYNQ image; report failures.
            file_fps = fps if fps > 0 else 60.0
            output_writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                file_fps, (1280, 720))
            if not output_writer.isOpened():
                raise RuntimeError('ERROR: Unable to write output file!')

        if width <= 0 or height <= 0:
            raise RuntimeError('ERROR: Invalid input video dimensions!')
        use_resize = (width != 1280) or (height != 720)
        if args.simulation is not None:
            sim_in = open(args.simulation + '.in', 'wb')
            sim_out = open(args.simulation + '.out', 'wb')

        print('Processing video data ...')
        HE = [WTHE(), WTHE()]
        HE_active = 0
        start = time.time()
        if profiler is not None:
            profiler.enable()

        while frames != args.max_frames:
            inframe = None
            try:
                if hdmi_in is not None:
                    inframe = hdmi_in.readframe()
                    frame = inframe
                    ret = True
                else:
                    ret, frame = input_capture.read()

                if not ret:
                    # HDMI readframe() blocks until it gets a frame, so EOF only
                    # applies to file input.
                    if args.loops == 0:
                        # Reopen rather than relying on seek support in the codec.
                        input_capture.release()
                        input_capture = cv2.VideoCapture(args.input)
                        if not input_capture.isOpened():
                            raise RuntimeError('ERROR: Unable to loop input file!')
                        continue
                    args.loops -= 1
                    if args.loops == 0:
                        break
                    input_capture.release()
                    input_capture = cv2.VideoCapture(args.input)
                    if not input_capture.isOpened():
                        raise RuntimeError('ERROR: Unable to reopen input file!')
                    continue

                if use_resize:
                    frame = cv2.resize(frame, (1280, 720))
                if sim_in is not None:
                    debug_dump(frame, sim_in)

                # Keep original WTHE processing and 4-frame histogram switching.
                if args.native:
                    frame = processing_native(frame, HE[HE_active])
                else:
                    frame = processing_numpy(frame, HE[HE_active])
                frames += 1
                if frames % 4 == 0:
                    if HE_thread is not None and HE_thread.is_alive():
                        print('WARNING: Waiting for HE update ...')
                        HE_thread.join()
                    HE_thread = threading.Thread(
                        target=HE[HE_active].transform_update, args=[frames])
                    HE_thread.start()
                    HE_active = not HE_active

                if hdmi_out is not None:
                    # Processing may change array dtype, shape and allocation.
                    # HDMI DMA requires a fresh PYNQ-compatible frame buffer.
                    outframe = hdmi_out.newframe()
                    try:
                        outframe[:] = frame
                        hdmi_out.writeframe(outframe)
                    except Exception:
                        outframe.freebuffer()
                        raise
                else:
                    output_writer.write(frame)
                if sim_out is not None:
                    debug_dump(frame, sim_out)
            finally:
                # Never leak an input DMA frame, even if processing fails.
                if inframe is not None:
                    inframe.freebuffer()

    except KeyboardInterrupt:
        print('Interrupted! Ending ...')
    except Exception as exc:
        error = exc
        print('ERROR:', exc)
    finally:
        if start is not None:
            if profiler is not None:
                profiler.disable()
            end = time.time()

        print('Cleaning up before end ...')
        if HE_thread is not None:
            HE_thread.join()
        if input_capture is not None:
            input_capture.release()
        if output_writer is not None:
            output_writer.release()
        if hdmi_in is not None:
            try:
                hdmi_in.stop()
            finally:
                hdmi_in.close()
        if hdmi_out is not None:
            try:
                hdmi_out.stop()
            finally:
                hdmi_out.close()
        if sim_in is not None:
            sim_in.close()
        if sim_out is not None:
            sim_out.close()

        if start is not None:
            duration = max(end - start, 1e-9)
            print()
            print(' Processing time:', '{:.3f}'.format(duration), 's')
            print('Processed frames:', frames)
            print('Processing speed:', '{:.3f}'.format(frames / duration), 'Fps')
            print('                 ', '{:.3f}'.format(
                (frames * 1280 * 720 / duration) / 1000000.0), 'Mpps')
            print()
        if profiler is not None and start is not None:
            profiler.dump_stats('video_ps.prof')
            print('Saved profiling data to video_ps.prof')
            pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(35)

    if error is not None:
        raise error


if __name__ == '__main__':
    main()
