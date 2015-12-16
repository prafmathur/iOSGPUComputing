//
//  Kernel.metal
//  iOSGPUComputing
//
//  Created by Praful Mathur on 11/8/15.
//  Copyright © 2015 Praf. All rights reserved.
//

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_integer>
#define MEANS 8
using namespace metal;

int difSq(int a, int b) {
    int i = absdiff(a,b);
    return i*i;
}



kernel void getRGB(const device int *pixel[[ buffer(0) ]],
                   device uint32_t *assignment [[buffer(1)]],
                   constant int *means [[buffer(2)]],
                   uint id [[ thread_position_in_grid ]]) {

    int pix = pixel[id];
    uint8_t R = (pix & 0x000000ff);
    uint8_t G = (pix & 0x0000ff00) >> 8;
    uint8_t B = (pix & 0x00ff0000) >> 16;
    uint8_t A = (pix & 0xff000000) >> 24;
    
    int k = MEANS;
    int distances[k];
    uint32_t assignedMean = 0;
    int minDistance = 2147483647;
    for(int i = 0; i < k; i++) {
        int mean = means[i];
        uint8_t meanR = (mean & 0x000000ff);
        uint8_t meanG = (mean & 0x0000ff00) >> 8;
        uint8_t meanB = (mean & 0x00ff0000) >> 16;
        uint8_t meanA = (mean & 0xff000000) >> 24;
        
        distances[i] = difSq(R, meanR) + difSq(G, meanG) + difSq(B, meanB) + difSq(A, meanA);
        if (distances[i] < minDistance) {
            minDistance = distances[i];
            assignedMean = i;
        }
    }
    assignment[id] = assignedMean;
    
}


kernel void updateMeans(const device int *pixels[[ buffer(0) ]],
                        const device int *assignments [[buffer(1)]],
                        device atomic_int *meanSums [[buffer(2)]],
                        device atomic_int *meanCounts [[buffer(3)]],
                        constant int *constants [[buffer(4)]],
                        uint id [[ thread_position_in_grid ]])
{
    
    int numPixels = constants[0];
    int numThreads = constants[2];
    int numThreadGroups = constants[3];
    
    int startPixel = id * numPixels/(numThreads*numThreadGroups);
    int endPixel = (id + 1) * numPixels/(numThreads*numThreadGroups);
    
    int localCounts[MEANS] = {0};
    int localSums[MEANS*4] = {0};
    
    for (int i = startPixel; i < endPixel; i++) {
        int pix = pixels[i];
        switch (assignments[i]){
            case 0:
                localCounts[0]++;
                localSums[0*4 + 0] += (pix & 0x000000ff);
                localSums[0*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[0*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[0*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 1:
                localCounts[1]++;
                localSums[1*4 + 0] += (pix & 0x000000ff);
                localSums[1*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[1*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[1*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 2:
                localCounts[2]++;
                localSums[2*4 + 0] += (pix & 0x000000ff);
                localSums[2*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[2*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[2*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 3:
                localCounts[3]++;
                localSums[3*4 + 0] += (pix & 0x000000ff);
                localSums[3*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[3*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[3*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 4:
                localCounts[4]++;
                localSums[4*4 + 0] += (pix & 0x000000ff);
                localSums[4*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[4*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[4*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 5:
                localCounts[5]++;
                localSums[5*4 + 0] += (pix & 0x000000ff);
                localSums[5*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[5*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[5*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 6:
                localCounts[6]++;
                localSums[6*4 + 0] += (pix & 0x000000ff);
                localSums[6*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[6*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[6*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 7:
                localCounts[7]++;
                localSums[7*4 + 0] += (pix & 0x000000ff);
                localSums[7*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[7*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[7*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 8:
                localCounts[8]++;
                localSums[8*4 + 0] += (pix & 0x000000ff);
                localSums[8*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[8*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[8*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 9:
                localCounts[9]++;
                localSums[9*4 + 0] += (pix & 0x000000ff);
                localSums[9*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[9*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[9*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 10:
                localCounts[10]++;
                localSums[10*4 + 0] += (pix & 0x000000ff);
                localSums[10*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[10*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[10*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 11:
                localCounts[11]++;
                localSums[11*4 + 0] += (pix & 0x000000ff);
                localSums[11*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[11*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[11*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 12:
                localCounts[12]++;
                localSums[12*4 + 0] += (pix & 0x000000ff);
                localSums[12*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[12*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[12*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 13:
                localCounts[13]++;
                localSums[13*4 + 0] += (pix & 0x000000ff);
                localSums[13*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[13*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[13*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 14:
                localCounts[14]++;
                localSums[14*4 + 0] += (pix & 0x000000ff);
                localSums[14*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[14*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[14*4 + 3] += (pix & 0xff000000) >> 24;
                break;
            case 15:
                localCounts[15]++;
                localSums[15*4 + 0] += (pix & 0x000000ff);
                localSums[15*4 + 1] += (pix & 0x0000ff00) >> 8;
                localSums[15*4 + 2] += (pix & 0x00ff0000) >> 16;
                localSums[15*4 + 3] += (pix & 0xff000000) >> 24;
                break;
//            case 16:
//                localCounts[16]++;
//                localSums[16*4 + 0] += (pix & 0x000000ff);
//                localSums[16*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[16*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[16*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 17:
//                localCounts[17]++;
//                localSums[17*4 + 0] += (pix & 0x000000ff);
//                localSums[17*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[17*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[17*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 18:
//                localCounts[18]++;
//                localSums[18*4 + 0] += (pix & 0x000000ff);
//                localSums[18*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[18*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[18*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 19:
//                localCounts[19]++;
//                localSums[19*4 + 0] += (pix & 0x000000ff);
//                localSums[19*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[19*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[19*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 20:
//                localCounts[20]++;
//                localSums[20*4 + 0] += (pix & 0x000000ff);
//                localSums[20*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[20*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[20*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 21:
//                localCounts[21]++;
//                localSums[21*4 + 0] += (pix & 0x000000ff);
//                localSums[21*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[21*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[21*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 22:
//                localCounts[22]++;
//                localSums[22*4 + 0] += (pix & 0x000000ff);
//                localSums[22*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[22*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[22*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 23:
//                localCounts[23]++;
//                localSums[23*4 + 0] += (pix & 0x000000ff);
//                localSums[23*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[23*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[23*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 24:
//                localCounts[24]++;
//                localSums[24*4 + 0] += (pix & 0x000000ff);
//                localSums[24*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[24*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[24*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 25:
//                localCounts[25]++;
//                localSums[25*4 + 0] += (pix & 0x000000ff);
//                localSums[25*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[25*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[25*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 26:
//                localCounts[26]++;
//                localSums[26*4 + 0] += (pix & 0x000000ff);
//                localSums[26*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[26*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[26*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 27:
//                localCounts[27]++;
//                localSums[27*4 + 0] += (pix & 0x000000ff);
//                localSums[27*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[27*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[27*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 28:
//                localCounts[28]++;
//                localSums[28*4 + 0] += (pix & 0x000000ff);
//                localSums[28*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[28*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[28*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 29:
//                localCounts[29]++;
//                localSums[29*4 + 0] += (pix & 0x000000ff);
//                localSums[29*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[29*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[29*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 30:
//                localCounts[30]++;
//                localSums[30*4 + 0] += (pix & 0x000000ff);
//                localSums[30*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[30*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[30*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
//            case 31:
//                localCounts[31]++;
//                localSums[31*4 + 0] += (pix & 0x000000ff);
//                localSums[31*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[31*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[31*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
            }
                
        }

    for (int i = 0; i < MEANS; i++) {
        atomic_fetch_add_explicit(&meanCounts[i], localCounts[i], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 0], localSums[i*4 + 0], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 1], localSums[i*4 + 1], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 2], localSums[i*4 + 2], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 3], localSums[i*4 + 3], memory_order_relaxed);
    }
}

kernel void updateImage(const device int *assignment [[buffer(0)]],
                   const device int *means [[buffer(1)]],
                   device int *outpixel[[ buffer(2) ]],
                   uint id [[ thread_position_in_grid ]]) {
    outpixel[id] = means[assignment[id]];
}
