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
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var defaultLibrary: MTLLibrary!
    var rgbProgram: MTLFunction!
    var updateMeans: MTLFunction!

    
    
    @IBAction func process(sender: UIButton) {
        metalImageProc()
        //kMeansSerial()
        //        testProc()
    }
//    var myvector = [Float](count: 10000000, repeatedValue: 0)
//    var finalResultArray = [Float](count: 10000000, repeatedValue: 0)

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

//    func serialComputation() {
//        let start = CACurrentMediaTime()
//
//        for (idx, val) in myvector.enumerate() {
//            finalResultArray[idx] = 1.0/(1.0 + pow(2.71828,val))
//        }
//        
//        let end = CACurrentMediaTime()
//        print(end - start)
//    }
    
    func kMeansSerial() {
        let start = CACurrentMediaTime()
        let img = picture.image?.CGImage!
        let pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        let imgdata: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let lengthOfData = CFDataGetLength(pixelData)
//        let width = Int((picture.image?.size.width)!)
//        let height = Int((picture.image?.size.height)!)
        
        let k = 8
        var means = [UInt8]()
        var assignments = [Int](count: lengthOfData/4, repeatedValue: 0)
        // Initialize k means to random values
        for _ in 0...k {
            let randomPixel = Int(arc4random_uniform(UInt32(lengthOfData/4))) * 4
            means.append(imgdata[randomPixel])
            means.append(imgdata[randomPixel + 1])
            means.append(imgdata[randomPixel + 2])
            means.append(imgdata[randomPixel + 3])
        }
        
        var distances = [Int]()
        for i in 0...lengthOfData/4 {
            var R = imgdata[i*4]
            var G = imgdata[i*4 + 1]
            var B = imgdata[i*4 + 2]
            var A = imgdata[i*4 + 3]
            
            for j in 0...k {
                var meanR = imgdata[j*4]
                var meanG = imgdata[j*4 + 1]
                var meanB = imgdata[j*4 + 2]
                var meanA = imgdata[j*4 + 3]
                var sum = difSq(R, b: meanR) + difSq(G, b: meanG) + difSq(B, b: meanB) + difSq(A, b: meanA)
                distances.append(sum)
            }

            var min = 0
            var minDist = distances[0]
            for l in 1...distances.count-1 {
                if distances[l] < minDist {
                    min = l
                    minDist = distances[l]
                }
            }
            
            assignments[i] = min
        }
        
        var serialCounts = [UInt32](count: k, repeatedValue: 0)
        for i in assignments {
            serialCounts[Int(i)]++
        }
        
        print(serialCounts)
        
        let end = CACurrentMediaTime()
        print(end - start)
        
        
    }
    
    func difSq(a: UInt8, b:UInt8) -> Int {
        let aInt = Int(a)
        let bInt = Int(b)
        return abs(aInt-bInt) * abs(aInt-bInt)
    }
    
    
    func metalImageProc() {
        let start = CACurrentMediaTime()
        let img = picture.image?.CGImage!
        let pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        let imgdata: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let lengthOfData = CFDataGetLength(pixelData)
//        let width = Int((picture.image?.size.width)!)
//        let height = Int((picture.image?.size.height)!)
//        
        let k = 8
        var means = [UInt8]()
        // Initialize k means to random values
        for _ in 0...k {
            let randomPixel = Int(arc4random_uniform(UInt32(lengthOfData/4))) * 4
            means.append(imgdata[randomPixel])
            means.append(imgdata[randomPixel + 1])
            means.append(imgdata[randomPixel + 2])
            means.append(imgdata[randomPixel + 3])
        }
        
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
    
    
    
    struct PixelData {
        var r: UInt8
        var g: UInt8
        var b: UInt8
        var a: UInt8
    }

    func imageFromARGB32Bitmap(pixels:[PixelData], width: Int, height: Int)-> UIImage {
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo:CGBitmapInfo = CGBitmapInfo()
        let bitsPerComponent:Int = 8
        let bitsPerPixel:Int = 32
        
        assert(pixels.count == Int(width * height))
        
        var data = pixels // Copy to mutable []
        let providerRef = CGDataProviderCreateWithCFData(
            NSData(bytes: &data, length: data.count * sizeof(PixelData))
        )
        
        let cgim = CGImageCreate(
            width,
            height,
            bitsPerComponent,
            bitsPerPixel,
            width * Int(sizeof(PixelData)),
            rgbColorSpace,
            bitmapInfo,
            providerRef,
            nil,
            true,
            CGColorRenderingIntent.RenderingIntentDefault
        )
        return UIImage(CGImage: cgim!)
    }
    
    func testProc() {
        let img = picture.image?.CGImage!
        
        let pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        let imgdata: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let lengthOfData = CFDataGetLength(pixelData)
        var p = [PixelData](count: Int(lengthOfData)/4, repeatedValue: PixelData(r: 0, g: 0, b: 0, a: 0))
        print(p.count)
        for idx in 0...p.count - 1 {
            let pix = idx * 4
            var rval = UInt8(imgdata[pix])
            if (rval > 5) {
                rval = rval - 5
            }
            p[idx] = PixelData(r: rval, g: imgdata[pix+1], b: imgdata[pix+2], a: imgdata[pix+3])
        }
        picture.image = imageFromARGB32Bitmap(p, width: Int((picture.image?.size.width)!), height: Int(picture.image!.size.height))
        
        
        //
//        let providerRef = CGDataProviderCreateWithCFData(
//            NSData(bytes: imgdata, length: lengthOfData)
//        )
//        
//        let width = Int((picture.image?.size.width)!)
//        let height = Int((picture.image?.size.height)!)
//        
//        let bitsPerComponent:Int = k
//        let bitsPerPixel:Int = 32
//        let cgim = CGImageCreate(
//            width,
//            height,
//            bitsPerComponent,
//            bitsPerPixel,
//            width * 4,
//            CGColorSpaceCreateDeviceRGB(),
//            CGBitmapInfo(rawValue: CGImageAlphaInfo.PremultipliedFirst.rawValue),
//            providerRef,
//            nil,
//            true,
//            CGColorRenderingIntent.RenderingIntentDefault
//        )
//        
//        let newImage = UIImage(CGImage: cgim!)
//        picture.image = newImage

        
    }
    
//    func metalComputation() {
//        let start = CACurrentMediaTime()
//
//        let device = MTLCreateSystemDefaultDevice()
//        let commandQueue = device?.newCommandQueue()
//        let defaultLibrary = device?.newDefaultLibrary()
//        let commandBuffer = commandQueue?.commandBuffer()
//        let computeCommandEncoder = commandBuffer?.computeCommandEncoder()
//
//        
//        let sigmoidProgram = defaultLibrary!.newFunctionWithName("sigmoid")
//        
//        
//        var computePipelineFilter: MTLComputePipelineState? = nil
//        do {
//          computePipelineFilter  = try device?.newComputePipelineStateWithFunction(sigmoidProgram!)
//        }
//        catch {
//            
//        }
//        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
//        let lengthOfVector = sizeofValue(myvector[0]) * myvector.count
//        let inBuffer = device?.newBufferWithBytes(&myvector, length: lengthOfVector, options: MTLResourceOptions())
//        computeCommandEncoder?.setBuffer(inBuffer, offset: 0, atIndex: 0)
//        
//        var resultdata = [Float](count:myvector.count, repeatedValue: 0)
//        let outVectorBuffer = device!.newBufferWithBytes(&resultdata, length: lengthOfVector, options: MTLResourceOptions())
//        computeCommandEncoder!.setBuffer(outVectorBuffer, offset: 0, atIndex: 1)
//        
//        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
//        let numThreadGroups = MTLSize(width: (myvector.count + 31)/32, height: 1, depth: 1)
//        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
//        computeCommandEncoder?.endEncoding()
//        
//        commandBuffer?.commit()
//        commandBuffer?.waitUntilCompleted()
//        
//        let data = NSData(bytesNoCopy: outVectorBuffer.contents(), length: myvector.count * sizeof(Float), freeWhenDone: false)
//        
//        
//        data.getBytes(&finalResultArray, length:myvector.count * sizeof(Float))
//        
//        let end = CACurrentMediaTime()
//        print(end - start)
//        
//    }

    
    
    //        let bitsPerComponent:Int = k
    //        let bitsPerPixel:Int = 32
    //        let providerRef = CGDataProviderCreateWithCFData(
    //            NSData(bytes: &resultData, length: lengthOfData)
    //        )
    //
    //
    //
    //        let cgim = CGImageCreate(
    //            width,
    //            height,
    //            bitsPerComponent,
    //            bitsPerPixel,
    //            width * 4,
    //            CGColorSpaceCreateDeviceRGB(),
    //            CGBitmapInfo(),
    //            providerRef,
    //            nil,
    //            true,
    //            CGColorRenderingIntent.RenderingIntentDefault
    //        )
    //
    //        let newImage = UIImage(CGImage: cgim!)
    //        picture.image = newImage
}

