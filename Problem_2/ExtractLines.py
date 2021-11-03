#!/usr/bin/env python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from numpy.lib.function_base import append
from PlotFunctions import *
from Utilities import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm.

    Inputs:
        RangeData: (x_r, y_r, theta, rho)
            x_r: robot's x position (m).
            y_r: robot's y position (m).
            theta: (1D) np array of angle 'theta' from data (rads).
            rho: (1D) np array of distance 'rho' from data (m).
        params: dictionary of parameters for line extraction.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        segend: np array (N_lines, 4) of line segment endpoints. Each row represents [x1, y1, x2, y2].
        pointIdx: (N_lines,2) segment's first and last point index.
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    startIdx = 0
    for endIdx in LineBreak:
        print('###   FITTING & SPLITTING   ###')
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
            print('###   MERGING   ###')
            alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    # # ### Filter Lines ###
    # # # Find and remove line segments that are too short
    # goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
    #                       (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    # pointIdx = pointIdx[goodSegIdx, :]
    # alpha = alpha[goodSegIdx]
    # r = r[goodSegIdx]
    # segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r

    return alpha, r, segend, pointIdx
############################################################
def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, which
    recursively sub-divides line segments until no further splitting is
    required.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        startIdx: starting index of segment to be split.
        endIdx: ending index of segment to be split.
        params: dictionary of parameters.
    Outputs:
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        idx: (N_lines,2) segment's first and last point index.

    HINT: Call FitLine() to fit individual line segments.
    HINT: Call FindSplit() to find an index to split at.
    '''
    ########## Code starts here #########
    #creat empty outputs
    r = np.zeros(0)
    alpha = np.zeros(0)
    idx = np.zeros((0, 2), dtype=np.int)

    print('start@: ', startIdx, '   end@: ', endIdx)

    #isolate sub section of theta and rho
    this_theta = idxFromTo(theta,startIdx,endIdx)
    this_rho = idxFromTo(rho,startIdx,endIdx)

    #fit line and grab alpha and r
    new_alpha, new_r = FitLine( this_theta, this_rho )
    ## new_idx this may need to be add start_idx to get absolute index
    new_idx = FindSplit( this_theta, this_rho, new_alpha, new_r, params )

    if new_idx != -1:
        print('Recursing')
        alpha_low, r_low, idx_low = SplitLinesRecursive( theta, rho, startIdx, startIdx + new_idx, params) 
        alpha_high, r_high, idx_high = SplitLinesRecursive( theta, rho, startIdx + new_idx + 1ß , endIdx, params)
        alpha = np.append(alpha, alpha_low)
        r = np.append(r, r_low)
        idx = np.vstack([idx, idx_low]) 
        alpha = np.append(alpha, alpha_high)
        r = np.append(r, r_high)
        idx = np.vstack([idx, idx_high])
    else:
        alpha = np.append(alpha, new_alpha)
        r = np.append(r, new_r)
        idx = np.vstack([idx, (startIdx,endIdx)])

    ########## Code ends here ##########
    return alpha, r, idx
############################################################
def FindSplit(theta, rho, alpha, r, params):
    '''
    This function takes in a line segment and outputs the best index at which to
    split the segment, or -1 if no split should be made.

    The best point to split at is the one whose distance from the line is
    the farthest, as long as this maximum distance exceeds
    LINE_POINT_DIST_THRESHOLD and also does not divide the line into segments
    smaller than MIN_POINTS_PER_SEGMENT. Otherwise, no split should be made.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: 'alpha' of input line segment (1 number).
        r: 'r' of input line segment (1 number).
        params: dictionary of parameters.
    Output:
        splitIdx: idx at which to split line (return -1 if it cannot be split).
    '''
    ########## Code starts here ##########
    n = len(theta)
    greatest_dist = 0
    this_dist = np.zeros(0)

    if n < (2*params['MIN_POINTS_PER_SEGMENT'])-1:
        splitIdx = -1
        ### TOBEFIXED need to account for case when max is at the last index
    else:
        for i in range(n):
            this_dist = np.append(this_dist, abs(np.cos( theta[i] - alpha ) * rho[i] - r))

        check_dist = removeEnds(this_dist,params['MIN_POINTS_PER_SEGMENT']-1,params['MIN_POINTS_PER_SEGMENT']-1)
        greatest_dist = np.max(check_dist)
        
        if greatest_dist < params['LINE_POINT_DIST_THRESHOLD']:
            splitIdx = -1
        else:
            if getEndIdx(check_dist) == np.where(check_dist==np.max(check_dist))[0][0]:
                splitIdx = params['MIN_POINTS_PER_SEGMENT']-1 + np.where(check_dist==np.max(check_dist))[0][0] -1
            else:
                splitIdx = params['MIN_POINTS_PER_SEGMENT']-1 + np.where(check_dist==np.max(check_dist))[0][0]

    ## for debugging
    if (splitIdx != -1):
        print('       split')
        print('       size: ',n,', index: ',splitIdx)
    else:
        print('       did not split')
    
    ########## Code ends here ##########

    return splitIdx
############################################################
def FitLine(theta, rho):
    '''
    This function outputs a least squares best fit line to a segment of range
    data, expressed in polar form (alpha, r).

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
    Outputs:
        alpha: 'alpha' of best fit for range data (1 number) (rads). Should be between -pi and pi.
        r: 'r' of best fit for range data (1 number) (m). Should be positive.
    '''
    ########## Code starts here ##########
    n = len(theta)
    A, B, C, D, E = 0,0,0,0,0
    for i in range(n):
        A = A + (rho[i]**2)*np.sin(2*theta[i])
        B = B + (rho[i]**2)*np.cos(2*theta[i])
        for j in range(n):
            C = C + rho[i]*rho[j]*np.cos(theta[i])*np.sin(theta[j])
            D = D + rho[i]*rho[j]*np.cos(theta[i]+theta[j])
    alpha = (1/2) * np.arctan2( (A-(2/n)*C) , (B-(1/n)*D) ) + (np.pi/2)

    for i in range(n):
        E = E + rho[i]*np.cos(theta[i]-alpha)

    r = (1/n)*E
    ########## Code ends here ##########
    return alpha, r
############################################################
def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    This function merges neighboring segments that are colinear and outputs a
    new set of line segments.

    Inputs:
        theta: (1D) np array of angle 'theta' from data (rads).
        rho: (1D) np array of distance 'rho' from data (m).
        alpha: (1D) np array of 'alpha' for each fitted line (rads).
        r: (1D) np array of 'r' for each fitted line (m).
        pointIdx: (N_lines,2) segment's first and last point indices.
        params: dictionary of parameters.
    Outputs:
        alphaOut: output 'alpha' of merged lines (rads).
        rOut: output 'r' of merged lines (m).
        pointIdxOut: output start and end indices of merged line segments.

    HINT: loop through line segments and try to fit a line to data points from
          two adjacent segments. If this line cannot be split, then accept the
          merge. If it can be split, do not merge.
    '''
    ########## Code starts here ##########
    # theta, rho, alpha, r, pointIdx, params
    alphaOut = np.zeros(0)              #create empty array to return new set of alphas
    rOut = np.zeros(0)                  #create empty array to return new set of r
    pointIdxOut = np.zeros((0,2), dtype = int)  
    
    n=len(alpha)
    i = 0
    
    while i < n-1:
            startindex = pointIdx[i,0]
            endindex = pointIdx[i+1,1]
            print('start@: ', startindex, '   end@: ', endindex)
            new_theta = idxFromTo(theta,startindex,endindex)
            new_rho = idxFromTo(rho,startindex,endindex)
            new_pointIdx = ([startindex, endindex])
            new_alpha, new_r = FitLine(new_theta,new_rho)
            merged = FindSplit(new_theta, new_rho, new_alpha, new_r, params)
            if -1 == merged:
                alphaOut = np.append(alphaOut, new_alpha)         #should these be single values?
                rOut = np.append(rOut, new_r)                     #should these be single values?
                pointIdxOut = np.vstack([pointIdxOut, new_pointIdx])   #should these be single values?
                print('       merge')
                i = i + 2
            else:
                #if can be split keep split and add both tented entries to list
                #pause here to check if alpha is of the same size as pointIdx
                new_alpha = (alpha[i]) 
                new_r = (r[i])
                alphaOut = np.append(alphaOut, new_alpha)         #should these be single values?
                rOut = np.append(rOut, new_r)                     #should these be single values?
                pointIdxOut = np.vstack([pointIdxOut, pointIdx[i]])
                #if this is second to last group and not combined then add last entry
                if i == n-2:
                    new_alpha = (alpha[i+1]) 
                    new_r = (r[i+1])
                    alphaOut = np.append(alphaOut, new_alpha)         #should these be single values?
                    rOut = np.append(rOut, new_r)                     #should these be single values?
                    pointIdxOut = np.vstack([pointIdxOut, pointIdx[i+1]])
                i = i + 1
    ########## Code ends here ##########
    return alphaOut, rOut, pointIdxOut

############################################################

#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = .1 #0.05 #for pose 1,3 #0.05 #for pose 2 # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = .07 #0.05 #for pose 1,3 #0.05 #for pose 2 # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 2 #2 #for pose 1,3 #3 #for pose 2 # minimum number of points per line segment
    MAX_P2P_DIST = 4 #.5 #for pose 1,3 #1 #for pose 2 # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'
    #filename = 'rangeData_4_9_360.csv'
    #filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show()

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
