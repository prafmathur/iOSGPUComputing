//
//  ViewController.swift
//  iOSGPUComputing
//
//  Created by Praful Mathur on 11/8/15.
//  Copyright © 2015 Praf. All rights reserved.
//

import UIKit
import CoreGraphics
import Metal
import Darwin



class ViewController: UIViewController {

    @IBOutlet weak var picture: UIImageView!
    
    let k = 8

    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var defaultLibrary: MTLLibrary!
    var rgbProgram: MTLFunction!
    var updateMeans: MTLFunction!
    var means = [UInt8]()
    var img : CGImage!
    var pixelData: CFData!
    var imgdata: UnsafePointer<UInt8>!
    var lengthOfData: CFIndex!

    
    
    @IBAction func process(sender: UIButton) {
        metalImageProc()

    }
    override func viewDidLoad() {
        super.viewDidLoad()
        
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.newCommandQueue()
        self.defaultLibrary = device?.newDefaultLibrary()
        self.rgbProgram = defaultLibrary!.newFunctionWithName("getRGB")
        self.updateMeans = defaultLibrary!.newFunctionWithName("updateMeans")

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    
    func difSq(a: UInt8, b:UInt8) -> Int {
        let aInt = Int(a)
        let bInt = Int(b)
        return abs(aInt-bInt) * abs(aInt-bInt)
    }
    
    func initMeans() {
        self.img = picture.image?.CGImage!
        self.pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        self.imgdata = CFDataGetBytePtr(pixelData)
        self.lengthOfData = CFDataGetLength(pixelData)
        // Initialize k means to random values
        for _ in 0...k {
            let randomPixel = Int(arc4random_uniform(UInt32(lengthOfData/4))) * 4
            means.append(imgdata[randomPixel])
            means.append(imgdata[randomPixel + 1])
            means.append(imgdata[randomPixel + 2])
            means.append(imgdata[randomPixel + 3])
        }
    }
    
    func metalImageProc() {
        let start = CACurrentMediaTime()

        
        var computePipelineFilter: MTLComputePipelineState? = nil
        do {
            computePipelineFilter  = try device?.newComputePipelineStateWithFunction(rgbProgram!)
        }
        catch {
            
        }
        let commandBuffer = commandQueue?.commandBuffer()
        let computeCommandEncoder = commandBuffer?.computeCommandEncoder()

        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        let inBuffer = device?.newBufferWithBytes(imgdata, length: lengthOfData, options: MTLResourceOptions())
        computeCommandEncoder!.setBuffer(inBuffer, offset: 0, atIndex: 0)

        print(lengthOfData)
        print(lengthOfData/4)
        
        var assignments = [UInt32](count: lengthOfData/4, repeatedValue: 0)
        let assignmentsBufferOut = device?.newBufferWithBytes(&assignments, length: assignments.count * sizeof(UInt32), options: MTLResourceOptions())
        computeCommandEncoder!.setBuffer(assignmentsBufferOut, offset: 0, atIndex: 1)
        
        let meansBuffer = device.newBufferWithBytes(&means, length: k*4, options: MTLResourceOptions())
        computeCommandEncoder?.setBuffer(meansBuffer, offset: 0, atIndex: 2)
        
        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: ((lengthOfData/4) + 31)/32, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()

        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()

        let data = NSData(bytesNoCopy: assignmentsBufferOut!.contents(), length: lengthOfData, freeWhenDone: false)
        data.getBytes(&assignments, length: lengthOfData/4)
        
        for i in 0...10 {
            print(assignments[i])
        }
        let middle = CACurrentMediaTime()
        print(middle - start)
        var serialCounts = [UInt32](count: k, repeatedValue: 0)
        for i in assignments {
            serialCounts[Int(i)]++
        }
        print(serialCounts)
        //---------------------------------
        var updateComputePipelineFilter: MTLComputePipelineState? = nil
        do {
            updateComputePipelineFilter  = try device?.newComputePipelineStateWithFunction(updateMeans!)
        }
        catch {
            
        }
        let updateCommandBuffer = commandQueue?.commandBuffer()
        let updateComputeCommandEncoder = updateCommandBuffer?.computeCommandEncoder()
        updateComputeCommandEncoder?.setComputePipelineState(updateComputePipelineFilter!)
        let imgForUpdate = device?.newBufferWithBytes(imgdata, length: lengthOfData, options: MTLResourceOptions())
        updateComputeCommandEncoder!.setBuffer(imgForUpdate, offset: 0, atIndex: 0)
        
        let assignmentsBuffer = device.newBufferWithBytes(&assignments, length: lengthOfData, options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(assignmentsBuffer, offset: 0, atIndex: 1)

        var meanSums = [UInt32](count: 32, repeatedValue: 0)
        let meanSumsBuffer = device.newBufferWithBytes(&meanSums, length: 32 * sizeof(UInt32), options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(meanSumsBuffer, offset: 0, atIndex: 2)
        
        var meanCounts = [UInt32](count: k, repeatedValue: 0)
        let meanCountsBuffer = device.newBufferWithBytes(&meanCounts, length: k * sizeof(UInt32), options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(meanCountsBuffer, offset: 0, atIndex: 3)
        
        updateComputeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        updateComputeCommandEncoder?.endEncoding()
        
        updateCommandBuffer?.commit()
        updateCommandBuffer?.waitUntilCompleted()
        
        let sumData = NSData(bytesNoCopy: meanSumsBuffer.contents(), length: 32 * sizeof(UInt32), freeWhenDone: false)
        sumData.getBytes(&meanSums, length: lengthOfData/4)
        
        let countData = NSData(bytesNoCopy: meanCountsBuffer.contents(), length: k * sizeof(UInt32), freeWhenDone: false)
        countData.getBytes(&meanCounts, length: lengthOfData/4)
        
        print(meanSums)
        print(meanCounts)
        

        print(means)
        
        for i in 0...8 {
            let updatedMean = meanSums[i]/meanCounts[i/4]
            means[i] = UInt8(updatedMean)
        }
        print(means)
        
        let end = CACurrentMediaTime()
        print(end - middle)

    }
    
}

