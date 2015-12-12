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

using namespace metal;


kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU
    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}

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
    
    int k = 8;
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
                        constant int *numPixels [[buffer(4)]],
                        uint id [[ thread_position_in_grid ]],
                        uint localid [[ thread_position_in_threadgroup]] )
{
    
    
    int startPixel = id * numPixels[0]/(32*16);
    int endPixel = (id + 1) * numPixels[0]/(32*16);
    
    int localCounts[8] = {0};
    int localSums[8*4] = {0};
    
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
//            case 8:
//                localCounts[8]++;
//                localSums[8*4 + 0] += (pix & 0x000000ff);
//                localSums[8*4 + 1] += (pix & 0x0000ff00) >> 8;
//                localSums[8*4 + 2] += (pix & 0x00ff0000) >> 16;
//                localSums[8*4 + 3] += (pix & 0xff000000) >> 24;
//                break;
            }
                
        }


    for (int i = 0; i < 8; i++) {
        atomic_fetch_add_explicit(&meanCounts[i], localCounts[i], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 0], localSums[i*4 + 0], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 1], localSums[i*4 + 1], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 2], localSums[i*4 + 2], memory_order_relaxed);
        atomic_fetch_add_explicit(&meanSums[i*4 + 3], localSums[i*4 + 3], memory_order_relaxed);
    }
    
    
}


//kernel void updateMeans(const device int *pixel[[ buffer(0) ]],
//                        const device int *assignment [[buffer(1)]],
//                        device atomic_int *meanSums [[buffer(2)]],
//                        device atomic_int *meanCounts [[buffer(3)]],
//                        uint id [[ thread_position_in_grid ]],
//                        uint localid [[ thread_position_in_threadgroup]] )
//{
//    int whichMean = assignment[id];
//    int pix = pixel[id];
//    int RGBA[4];
//    
//    
//    
//    
//    
////    RGBA[0] = (pix & 0x000000ff);
////    RGBA[1] = (pix & 0x0000ff00) >> 8;
////    RGBA[2] = (pix & 0x00ff0000) >> 16;
////    RGBA[3] = (pix & 0xff000000) >> 24;
////    
////    
////    atomic_fetch_add_explicit(&meanCounts[whichMean], 1, memory_order_relaxed);
////    atomic_fetch_add_explicit(&meanSums[whichMean*4 + 0], RGBA[0], memory_order_relaxed);
////    atomic_fetch_add_explicit(&meanSums[whichMean*4 + 1], RGBA[1], memory_order_relaxed);
////    atomic_fetch_add_explicit(&meanSums[whichMean*4 + 2], RGBA[2], memory_order_relaxed);
////    atomic_fetch_add_explicit(&meanSums[whichMean*4 + 3], RGBA[3], memory_order_relaxed);
//    
//}


kernel void updateImage(const device int *assignment [[buffer(0)]],
                   const device int *means [[buffer(1)]],
                   device int *outpixel[[ buffer(2) ]],
                   uint id [[ thread_position_in_grid ]]) {
    outpixel[id] = means[assignment[id]];
}
