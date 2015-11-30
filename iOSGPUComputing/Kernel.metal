//
//  Kernel.metal
//  iOSGPUComputing
//
//  Created by Praful Mathur on 11/8/15.
//  Copyright © 2015 Praf. All rights reserved.
//

#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;


kernel void sigmoid(const device float *inVector [[ buffer(0) ]],
                    device float *outVector [[ buffer(1) ]],
                    uint id [[ thread_position_in_grid ]]) {
    // This calculates sigmoid for _one_ position (=id) in a vector per call on the GPU
    outVector[id] = 1.0 / (1.0 + exp(-inVector[id]));
}

int difSq(int a, int b) {
    int absval = max(a,b) - min(a, b);
    return absval * absval;
}



kernel void getRGB(const device int *pixel[[ buffer(0) ]],
                   device uint32_t *assignment [[buffer(1)]],
                   const device int *means [[buffer(2)]],
                   uint id [[ thread_position_in_grid ]]) {

    int pix = pixel[id];
    uint8_t R = (pix & 0x000000ff);
    uint8_t G = (pix & 0x0000ff00) >> 8;
    uint8_t B = (pix & 0x00ff0000) >> 16;
    uint8_t A = (pix & 0xff000000) >> 24;
    
    int k = 5;
    int distances[k];
    
    for(int i = 0; i < k; i++) {
        int mean = means[i];
        uint8_t meanR = (mean & 0x000000ff);
        uint8_t meanG = (mean & 0x0000ff00) >> 8;
        uint8_t meanB = (mean & 0x00ff0000) >> 16;
        uint8_t meanA = (mean & 0xff000000) >> 24;
        
        distances[i] = difSq(R, meanR) + difSq(G, meanG) + difSq(B, meanB) + difSq(A, meanA);
        
    }
    
    uint32_t assignedMean = 0;
    int minDistance = distances[0];
    for (int i = 1; i < k; ++i) {
        if (distances[i] < minDistance) {
            minDistance = distances[i];
            assignedMean = i;
        }
    }
//    int out = A << 24 | B << 16 | G << 8 | R;
    assignment[id] = assignedMean;
    
}

kernel void updateMeans(const device int *pixel[[ buffer(0) ]],
                        const device int *assignment [[buffer(1)]],
                        device atomic_int *meanSums [[buffer(2)]],
                        device atomic_int *meanCounts [[buffer(3)]],
                        uint id [[ thread_position_in_grid ]])
{
    int idx = assignment[id];
    int pix = pixel[id];
    atomic_fetch_add_explicit(&meanCounts[idx], 1, memory_order_relaxed);//    uint8_t RGBA[4];

    int RGBA[4];
    RGBA[0] = (pix & 0x000000ff);
    RGBA[1] = (pix & 0x0000ff00) >> 8;
    RGBA[2] = (pix & 0x00ff0000) >> 16;
    RGBA[3] = (pix & 0xff000000) >> 24;
    
    for(int i = 0; i < 4; ++i) {
        //meanSums[idx*4 + i] += RGBA[i];
        atomic_fetch_add_explicit(&meanSums[idx*4 + i], RGBA[i], memory_order_relaxed);
    }
}


kernel void updateImage(const device int *assignment [[buffer(0)]],
                   const device int *means [[buffer(1)]],
                   device int *outpixel[[ buffer(2) ]],
                   uint id [[ thread_position_in_grid ]]) {
    outpixel[id] = means[assignment[id]];
}
