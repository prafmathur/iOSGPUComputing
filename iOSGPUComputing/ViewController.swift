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



class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var picture: UIImageView!
    let imagePicker = UIImagePickerController()
    
    let k = 6

    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var defaultLibrary: MTLLibrary!
    var rgbProgram: MTLFunction!
    var updateMeansMtl: MTLFunction!
    var updateImgMtl: MTLFunction!
    var means = [UInt8]()
    var assignments: [UInt32]!
    var img : CGImage!
    var pixelData: CFData!
    var imgdata: UnsafePointer<UInt8>!
    var lengthOfData: CFIndex!

    
    
    @IBAction func process(sender: UIButton) {
        initMeans()
        for _ in 0...5{
            computeDistances()
            updateMeansSerial()
        }
        updateImage()
        
        
        //        updateMeans()
        //        testImageChange()
    }
    
    @IBAction func loadImage(sender: AnyObject) {
        presentViewController(imagePicker, animated: true, completion: nil)
    }
    
    
    func imagePickerController(picker: UIImagePickerController, didFinishPickingImage image: UIImage, editingInfo: [String : AnyObject]?) {
        picture.image = image
        
        dismissViewControllerAnimated(true, completion: nil)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = device?.newCommandQueue()
        self.defaultLibrary = device?.newDefaultLibrary()
        self.rgbProgram = defaultLibrary!.newFunctionWithName("getRGB")
        self.updateMeansMtl = defaultLibrary!.newFunctionWithName("updateMeans")
        self.updateImgMtl = defaultLibrary!.newFunctionWithName("updateImage")
        
        let width = Int((picture.image?.size.width)!)
        let height = Int((picture.image?.size.height)!)
        print(width)
        print(height)

        imagePicker.delegate = self
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
        self.assignments = [UInt32](count: lengthOfData/4, repeatedValue: 0)
    }
    
    
    func computeDistances() {
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
        var assign = [UInt32](count: lengthOfData/4, repeatedValue: 0)
        let assignmentsBufferOut = device?.newBufferWithBytes(&assign, length: assignments.count * sizeof(UInt32), options: MTLResourceOptions())
        computeCommandEncoder!.setBuffer(assignmentsBufferOut, offset: 0, atIndex: 1)
        
        let meansBuffer = device.newBufferWithBytes(&means, length: k*4, options: MTLResourceOptions())
        computeCommandEncoder?.setBuffer(meansBuffer, offset: 0, atIndex: 2)
        
        var kLocal = [k]
        let kBuffer = device.newBufferWithBytes(&kLocal, length: sizeof(Int), options: MTLResourceOptions())
        computeCommandEncoder?.setBuffer(kBuffer, offset: 0, atIndex: 3)
        
//    computeCommandEncoder?.setBytes(&kLocal, length: sizeof(Int), atIndex: 3)
        
        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: ((lengthOfData/4) + 31)/32, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()
        
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        
        let data = NSData(bytesNoCopy: assignmentsBufferOut!.contents(), length: assignments.count * sizeof(UInt32), freeWhenDone: false)
        data.getBytes(&assign, length: assignments.count * sizeof(UInt32))
        assignments = assign
        
        
        
        let middle = CACurrentMediaTime()
        print(middle - start)

    }
    
    func updateMeans() {
        let middle = CACurrentMediaTime()

        var updateComputePipelineFilter: MTLComputePipelineState? = nil
        do {
            updateComputePipelineFilter  = try device?.newComputePipelineStateWithFunction(updateMeansMtl!)
        }
        catch {
            
        }
        let updateCommandBuffer = commandQueue?.commandBuffer()
        let updateComputeCommandEncoder = updateCommandBuffer?.computeCommandEncoder()
        updateComputeCommandEncoder?.setComputePipelineState(updateComputePipelineFilter!)
        let imgForUpdate = device?.newBufferWithBytes(imgdata, length: lengthOfData, options: MTLResourceOptions())
        updateComputeCommandEncoder!.setBuffer(imgForUpdate, offset: 0, atIndex: 0)
        
        var assign = [UInt32](count: lengthOfData/4, repeatedValue: 0)
        assign = assignments
        let assignmentsBuffer = device.newBufferWithBytes(&assign, length: assign.count * sizeof(UInt32), options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(assignmentsBuffer, offset: 0, atIndex: 1)

        var meanSums = [UInt32](count: 32, repeatedValue: 0)
        let meanSumsBuffer = device.newBufferWithBytes(&meanSums, length: 32 * sizeof(UInt32), options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(meanSumsBuffer, offset: 0, atIndex: 2)
        
        var meanCounts = [UInt32](count: k, repeatedValue: 0)
        let meanCountsBuffer = device.newBufferWithBytes(&meanCounts, length: k * sizeof(UInt32), options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(meanCountsBuffer, offset: 0, atIndex: 3)
        
        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: ((lengthOfData/4) + 31)/32, height: 1, depth: 1)
        updateComputeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        updateComputeCommandEncoder?.endEncoding()
        
        updateCommandBuffer?.commit()
        updateCommandBuffer?.waitUntilCompleted()
        
        let sumData = NSData(bytesNoCopy: meanSumsBuffer.contents(), length: 32 * sizeof(UInt32), freeWhenDone: false)
        sumData.getBytes(&meanSums, length: lengthOfData/4)
        
        let countData = NSData(bytesNoCopy: meanCountsBuffer.contents(), length: k * sizeof(UInt32), freeWhenDone: false)
        countData.getBytes(&meanCounts, length: lengthOfData/4)
        
//        print(meanSums)
//        print(meanCounts)
        

//        print(means)
//        
//        for i in 0...8 {
//            let updatedMean = meanSums[i]/meanCounts[i/4]
//            means[i] = UInt8(updatedMean)
//        }
//        print(means)
        
        let end = CACurrentMediaTime()
        print(end - middle)

    }

    func updateImage() {
        
        var updateComputePipelineFilter: MTLComputePipelineState? = nil
        do {
            updateComputePipelineFilter  = try device?.newComputePipelineStateWithFunction(updateImgMtl!)
        }
        catch {
            
        }
        let updateCommandBuffer = commandQueue?.commandBuffer()
        let updateComputeCommandEncoder = updateCommandBuffer?.computeCommandEncoder()
        updateComputeCommandEncoder?.setComputePipelineState(updateComputePipelineFilter!)

        
        var assign = [UInt32](count: lengthOfData/4, repeatedValue: 0)
        assign = assignments
        let assignmentsBuffer = device.newBufferWithBytes(&assign, length: lengthOfData, options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(assignmentsBuffer, offset: 0, atIndex: 0)
        
        let meansBuffer = device.newBufferWithBytes(&means, length: k*4, options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(meansBuffer, offset: 0, atIndex: 1)
        
        var resultData = [UInt8](count: lengthOfData, repeatedValue: 0)
        let imgBuffer = device?.newBufferWithBytes(&resultData, length: lengthOfData, options: MTLResourceOptions())
        updateComputeCommandEncoder?.setBuffer(imgBuffer, offset: 0, atIndex: 2)
        
        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: ((lengthOfData/4) + 31)/32, height: 1, depth: 1)
        updateComputeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        updateComputeCommandEncoder?.endEncoding()
        
        updateCommandBuffer?.commit()
        updateCommandBuffer?.waitUntilCompleted()
        let data = NSData(bytesNoCopy: imgBuffer!.contents(), length: lengthOfData, freeWhenDone: false)
        data.getBytes(&resultData, length: lengthOfData)
    
        
        
        let width = Int((picture.image?.size.width)!)
        let height = Int((picture.image?.size.height)!)
        let bitsPerComponent:Int = 8
        let bitsPerPixel:Int = 32
        let providerRef = CGDataProviderCreateWithCFData(CFDataCreate(nil, &resultData, lengthOfData))

        let cgim = CGImageCreate(
            width,
            height,
            bitsPerComponent,
            bitsPerPixel,
            width * 4,
            CGColorSpaceCreateDeviceRGB(),
            CGBitmapInfo(),
            providerRef,
            nil,
            true,
            CGColorRenderingIntent.RenderingIntentDefault
        )
        
        let newImage = UIImage(CGImage: cgim!)
        picture.image = newImage
    }
    
    
    func updateMeansSerial() {
        let middle = CACurrentMediaTime()
        var serialCounts = [UInt32](count: k, repeatedValue: 0)
        for i in assignments {
            serialCounts[Int(i)]++
        }
        print(serialCounts)
    
        var meanSums = [UInt32](count: k*4, repeatedValue: 0)
    
        for i in 0...assignments.count - 1 {
            
            for j in 0...3 {
                meanSums[Int(assignments[i]) * 4 + j] += UInt32(imgdata[i*4+j])
            }
            
        }

//        print(meanSums)

//        print("Means Before \(means)")
//        
//        for i in 0...8 {
//            let updatedMean = meanSums[i]/serialCounts[i/4]
//            means[i] = UInt8(updatedMean)
//        }
//        
//        
//        print("Means after \(means)")
        let end = CACurrentMediaTime()
        print(end - middle)
    }
    
    
    
    func testImageChange() {
        let width = Int((picture.image?.size.width)!)
        let height = Int((picture.image?.size.height)!)
        let bitsPerComponent:Int = 8
        let bitsPerPixel:Int = 32
        let providerRef = CGDataProviderCreateWithCFData(
            NSData(bytes: imgdata, length: lengthOfData)
        )
        
        let cgim = CGImageCreate(
            width,
            height,
            bitsPerComponent,
            bitsPerPixel,
            width * 4,
            CGColorSpaceCreateDeviceRGB(),
            CGBitmapInfo(),
            providerRef,
            nil,
            true,
            CGColorRenderingIntent.RenderingIntentDefault
        )
        
        let newImage = UIImage(CGImage: cgim!)
        picture.image = newImage
    }
    

}

