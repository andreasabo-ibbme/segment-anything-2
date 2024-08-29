# importing the module 
import cv2 
import os, glob
import pandas as pd
from icecream import ic

pts = []
img = 0
# function to display the coordinates of 
# of the points clicked on the image  
def click_event(event, x, y, flags, params): 
    # checking for left mouse clicks 
    global pts

    if event == cv2.EVENT_LBUTTONDOWN: 
  
        # displaying the coordinates 
        # on the Shell 
        # print(x, ' ', y) 
        pts.append([x, y])
  
        # displaying the coordinates 
        # on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font, 
                    1, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
  
def process_image(input_path, output_file):
    global pts
    pts = []
    
    # reading the image 
    global img
    img = cv2.imread(input_path, 1) 
  
    # displaying the image 
    cv2.imshow('image', img) 
  
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv2.setMouseCallback('image', click_event) 
  
    # wait for a key to be pressed to exit 
    cv2.waitKey(0) 
  
    # close the window 
    cv2.destroyAllWindows() 
    print(f"POINTs", pts)
    df = pd.DataFrame(pts, columns=["x", "y"])

    df.to_csv(output_file)


def process_all(input_folder, FORCE_REPROCESS):
    all_subfolders = glob.glob(os.path.join(input_folder, "andrea*"))
    all_subfolders.sort()
    for folder in all_subfolders:
        image_file = glob.glob(os.path.join(folder, '*.png'))[0]
        output_file = os.path.join(os.path.dirname(image_file), "pts_for_SAM.csv")
        if not FORCE_REPROCESS and os.path.exists(output_file):
            continue
        
        ic(image_file, output_file)
        process_image(image_file, output_file)
    
    
# driver function 
if __name__=="__main__": 
    IMAGE = r"N:\AMBIENT\Andrea_S\EDS\DLC_working_dir\dlc_projects_participants\EDS001_EDS100__Thumb-liam-2023-07-10_SAM2\labeled-data\andrea-thumb_l__EDS001__thumb_l\img082.png"
    input_folder = r"N:\AMBIENT\Andrea_S\EDS\DLC_working_dir\dlc_projects_participants\EDS001_EDS100__Thumb-liam-2023-07-10_SAM2\labeled-data"
    FORCE_REPROCESS = False
    process_all(input_folder, FORCE_REPROCESS)
