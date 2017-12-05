import numpy as np
import cv2
import pdb
video_path = "/home/student-5/Downloads/stabilized.avi"
REF_FRAME_COUNT = 5


def blur(img, mask = None):
    blur_img = cv2.GaussianBlur(img, (25, 25), 0)

    if mask is None:
        return blur_img

    else:
        a = mask *  img + (1-mask) * blur_img
        return a.astype(np.uint8)


def get_ref_hsv_frames(cap, count):
    if not cap.isOpened():
        print "Cannot open the video file"
        return -1

    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_count - count)

    ref_frames = []
    while(cap.isOpened() and count > 0):
        ret, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ref_frames.append(hsv_frame )
        count -= 1

    return np.array(ref_frames)  # shape is count X width X height X 3



def initialize(frames):
    return np.median(frames, axis=0).astype(np.uint8)



def running(frame, ref_frame, th, center, max_size=None):
    """

    :param frame: hsv frame
    :param ref_frame:
    :param th:
    :param center:
    :param max_size:
    :return:
    """
    # compute the difference
    frame = frame.astype(int)
    ref_frame = ref_frame.astype(int)

    # compute the distance
    diff = frame - ref_frame

    H = diff[:,:,0]
    S = diff[:,:,1]

    dist = np.sqrt(np.square(H) + np.square(S)) > th
    dist = dist.astype(np.uint8)

    # compute connected components

    ret, labeled_comp = cv2.connectedComponents(dist)

    mask = None
    found = False
    print "num of components", ret, set(labeled_comp.flatten().tolist())
    if labeled_comp[center] != 0:  # background
        comp_id = labeled_comp[center]
        mask = labeled_comp == comp_id
        mask = np.repeat(mask, 3).reshape((mask.shape[0], mask.shape[1], 3))
        print sum(sum(labeled_comp==labeled_comp[center]))
        print "Found!!"
        found = True

    frame = frame.astype(np.uint8)
    return blur(frame, mask), found




def main(video_path, threshold, center= None):
    cap = cv2.VideoCapture(video_path)
    ref_frames = get_ref_hsv_frames(cap, REF_FRAME_COUNT)
    ref_median = initialize(ref_frames)
    if center is None:
        center = (ref_median.shape[0]/2, ref_median.shape[1]/2)



    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # return to the begining of the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        main_frame, found = running(hsv_frame, ref_median, threshold, center)

        main_frame = cv2.cvtColor(main_frame, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame',main_frame )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(video_path, 50)
