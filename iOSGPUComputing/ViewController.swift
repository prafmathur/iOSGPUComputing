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

    // Image View
    @IBOutlet weak var picture: UIImageView!

    // Number of means
    let k = 8
    
    // Metal GPU Objects
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var defaultLibrary: MTLLibrary!
    
    // Metal Functions
    var rgbProgram: MTLFunction!
    var updateMeansMtl: MTLFunction!
    var updateImgMtl: MTLFunction!
    
    // Common data structures
    var means = [UInt8]()
    var assignments: [UInt32]!
    var img : CGImage!
    var pixelData: CFData!
    var imgdata: UnsafePointer<UInt8>!
    var lengthOfData: CFIndex!
    var numPixelsInImage: Int!
    var inBuffer: MTLBuffer!
    var meansInit = false
    
    // Image Picker
    let imagePicker = UIImagePickerController()
    
    // Number of threads to run on GPU
    let numGPUthreads = 32;
    let threadGroupNumForUpdate = 8;
    
    
    @IBAction func process(sender: UIButton) {
        let START = CACurrentMediaTime()
        initMeans()
        for _ in 0...10 {
            computeDistances()
            updateMeans()
        }
        updateImage()
        let END = CACurrentMediaTime()
        print("Total Time for 10 iterations: \(END - START)")
        
        
    }
    
    
    
    @IBAction func saveImage(sender: AnyObject) {
        UIImageWriteToSavedPhotosAlbum(picture.image!, self, nil, nil)

    }
    
    @IBAction func iterate(sender: AnyObject) {
        let START = CACurrentMediaTime()
        if meansInit == false {
            initMeans()
        }
        let A = CACurrentMediaTime()
        computeDistances()
        let B = CACurrentMediaTime()
        updateMeans()
        let C = CACurrentMediaTime()
        updateImage()
        let END = CACurrentMediaTime()
        
        print(A - START)
        print(B - A)
        print(C - B)
        print(END - C)
        print(END - START)
    }
    
    @IBAction func loadImage(sender: AnyObject) {
        meansInit = false
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
        imagePicker.delegate = self
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func initMeans() {
        self.img = picture.image?.CGImage!
        self.pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        self.imgdata = CFDataGetBytePtr(pixelData)
        self.lengthOfData = CFDataGetLength(pixelData)
        self.numPixelsInImage = lengthOfData/4
        self.means = [UInt8]()
        // Initialize k means to random values
        for _ in 0...k {
            let randomPixel = Int(arc4random_uniform(UInt32(numPixelsInImage))) * 4
            means.append(imgdata[randomPixel])
            means.append(imgdata[randomPixel + 1])
            means.append(imgdata[randomPixel + 2])
            means.append(imgdata[randomPixel + 3])
        }
        self.assignments = [UInt32](count: numPixelsInImage, repeatedValue: 0)
        self.inBuffer = device?.newBufferWithBytes(imgdata, length: lengthOfData, options: MTLResourceOptions())
        meansInit = true
    }
    
    
    func computeDistances() {
        let start = CACurrentMediaTime()
        
        // --------- Initialize Command Pipeline --------------------------------
        var computePipelineFilter: MTLComputePipelineState? = nil
        do {
            computePipelineFilter  = try device?.newComputePipelineStateWithFunction(rgbProgram!)
        }
        catch {
            print("Error creating pipeline state")
            return
        }
        let commandBuffer = commandQueue?.commandBuffer()
        let computeCommandEncoder = commandBuffer?.computeCommandEncoder()
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        //------------------------------------------------------------------------

        //------------ Set Buffers ------------------------------------------------
        var assign = [UInt32](count: numPixelsInImage, repeatedValue: 0)
        let assignmentsBufferOut = device?.newBufferWithBytes(&assign, length: assignments.count * sizeof(UInt32), options: MTLResourceOptions.StorageModeShared)
        let meansBuffer = device.newBufferWithBytes(&means, length: k*4, options: MTLResourceOptions.StorageModeShared)
        computeCommandEncoder!.setBuffer(inBuffer, offset: 0, atIndex: 0)
        computeCommandEncoder!.setBuffer(assignmentsBufferOut, offset: 0, atIndex: 1)
        computeCommandEncoder!.setBuffer(meansBuffer, offset: 0, atIndex: 2)
        // -----------------------------------------------------------------------------
    
        let threadsPerGroup = MTLSize(width: numGPUthreads, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: (numPixelsInImage + (numGPUthreads-1))/numGPUthreads, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()
        
        let a = CACurrentMediaTime()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        let b = CACurrentMediaTime()
//        print("GPU Time For Distances \(b - a)")
        
        let data = NSData(bytesNoCopy: assignmentsBufferOut!.contents(), length: assignments.count * sizeof(UInt32), freeWhenDone: false)
        data.getBytes(&assign, length: assignments.count * sizeof(UInt32))
        assignments = assign
        
        let end = CACurrentMediaTime()
//        print("Total Time for Distances \(end - start)")
    }
    
    func updateMeans() {
        let middle = CACurrentMediaTime()
        
        var computePipelineFilter: MTLComputePipelineState? = nil
        do {
            computePipelineFilter  = try device?.newComputePipelineStateWithFunction(updateMeansMtl!)
        }
        catch let error as NSError {
            print(error.localizedDescription)
        }
        let commandBuffer = commandQueue?.commandBuffer()
        let computeCommandEncoder = commandBuffer?.computeCommandEncoder()
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        
        var assign = [UInt32](count: numPixelsInImage, repeatedValue: 0)
        assign = assignments
        var meanSums = [UInt32](count: k*4, repeatedValue: 0)
        var meanCounts = [UInt32](count: k, repeatedValue: 0)
        var constants = [UInt32(numPixelsInImage), UInt32(k), UInt32(numGPUthreads), UInt32(threadGroupNumForUpdate)]
        
        let assignmentsBuffer = device.newBufferWithBytes(&assign, length: assign.count * sizeof(UInt32), options: MTLResourceOptions.StorageModeShared)
        let meanSumsBuffer = device.newBufferWithBytes(&meanSums, length: k * 4 * sizeof(UInt32), options: MTLResourceOptions.StorageModeShared)
        let meanCountsBuffer = device.newBufferWithBytes(&meanCounts, length: k * sizeof(UInt32), options: MTLResourceOptions.StorageModeShared)
        let constantsBuffer = device.newBufferWithBytes(&constants, length: constants.count * sizeof(UInt32), options: MTLResourceOptions.StorageModeShared)

        computeCommandEncoder?.setBuffer(inBuffer, offset: 0, atIndex: 0)
        computeCommandEncoder?.setBuffer(assignmentsBuffer, offset: 0, atIndex: 1)
        computeCommandEncoder?.setBuffer(meanSumsBuffer, offset: 0, atIndex: 2)
        computeCommandEncoder?.setBuffer(meanCountsBuffer, offset: 0, atIndex: 3)
        computeCommandEncoder?.setBuffer(constantsBuffer, offset: 0, atIndex: 4)
        
        let threadsPerGroup = MTLSize(width: numGPUthreads, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: threadGroupNumForUpdate, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()

        let a = CACurrentMediaTime()
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        let b = CACurrentMediaTime()
//        print("GPU Time For Means\(b - a)")
        
//        let z = CACurrentMediaTime()

        let sumData = NSData(bytesNoCopy: meanSumsBuffer.contents(), length: k * 4 * sizeof(UInt32), freeWhenDone: false)
        sumData.getBytes(&meanSums, length: k * 4 * sizeof(UInt32))
        
        let countData = NSData(bytesNoCopy: meanCountsBuffer.contents(), length: k * sizeof(UInt32), freeWhenDone: false)
        countData.getBytes(&meanCounts, length: k * sizeof(UInt32))
//        
//        let zz = CACurrentMediaTime()
//        print(zz - z)
        for i in 0..<k {
            if meanCounts[i] != 0 {
                for j in 0...2 {
                    means[i*4+j] = UInt8(meanSums[i*4+j] / meanCounts[i])
                }
            }
            else {
                let randomPixel = Int(arc4random_uniform(UInt32(numPixelsInImage))) * 4
                for j in 0...3 {
                    means[i*4+j] = imgdata[randomPixel + j]
                }
            }
        }
//        let end = CACurrentMediaTime()
//        print("Total Time for Means \(end - middle)")

    }

    func updateImage() {
        var computePipelineFilter: MTLComputePipelineState? = nil
        do {
            computePipelineFilter  = try device?.newComputePipelineStateWithFunction(updateImgMtl!)
        }
        catch {
            
        }
        let commandBuffer = commandQueue?.commandBuffer()
        let computeCommandEncoder = commandBuffer?.computeCommandEncoder()
        computeCommandEncoder?.setComputePipelineState(computePipelineFilter!)
        
        var assign = [UInt32](count: numPixelsInImage, repeatedValue: 0)
        assign = assignments
        var resultData = [UInt8](count: lengthOfData, repeatedValue: 0)

        let assignmentsBuffer = device.newBufferWithBytes(&assign, length: lengthOfData, options: MTLResourceOptions.StorageModeShared)
        let meansBuffer = device.newBufferWithBytes(&means, length: k*4, options: MTLResourceOptions.StorageModeShared)
        let imgBuffer = device?.newBufferWithBytes(&resultData, length: lengthOfData, options: MTLResourceOptions.StorageModeShared)
        
        computeCommandEncoder?.setBuffer(assignmentsBuffer, offset: 0, atIndex: 0)
        computeCommandEncoder?.setBuffer(meansBuffer, offset: 0, atIndex: 1)
        computeCommandEncoder?.setBuffer(imgBuffer, offset: 0, atIndex: 2)
        
        let threadsPerGroup = MTLSize(width: 32, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: ((numPixelsInImage) + 31)/32, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()
        
        commandBuffer?.commit()
        commandBuffer?.waitUntilCompleted()
        
        let data = NSData(bytesNoCopy: imgBuffer!.contents(), length: lengthOfData, freeWhenDone: false)
        data.getBytes(&resultData, length: lengthOfData)
    
        let w = Int((picture.image?.size.width)!)
        let h = Int((picture.image?.size.height)!)
        let width = max(w,h)
        let height = min(w,h)
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
        
        dispatch_async(dispatch_get_main_queue(), {
            self.picture.image = newImage
            print("Width: \(w) Height: \(h)")
            print("Number of pixels in image \(self.numPixelsInImage)")

        })
        
    }
    
    func updateMeansSerial() {
        let middle = CACurrentMediaTime()
        var serialCounts = [UInt32](count: k, repeatedValue: 0)
        for i in assignments {
            serialCounts[Int(i)]++
        }
        var meanSums = [UInt32](count: k*4, repeatedValue: 0)
        
        for i in 0...assignments.count - 1 {
            for j in 0...3 {
                meanSums[Int(assignments[i]) * 4 + j] += UInt32(imgdata[i*4+j])
            }
        }
        
        for i in 0..<k {
            if serialCounts[i] != 0 {
                for j in 0...3 {
                    means[i*4+j] = UInt8(meanSums[i*4+j] / serialCounts[i])
                }
            }
            else {
                let randomPixel = Int(arc4random_uniform(UInt32(numPixelsInImage))) * 4
                for j in 0...3 {
                    means[i*4+j] = imgdata[randomPixel + j]
                }
            }
        }
        let end = CACurrentMediaTime()
        print(end - middle)
        
    }
    
    func computeDistanceSerial() {
        let middle = CACurrentMediaTime()
        var r,g,b,a,w,x,y,z: Int!
        var dist, minDist, closestMean: Int!
        for i in 0..<numPixelsInImage {
            r = Int(imgdata[i*4])
            g = Int(imgdata[i*4 + 1])
            b = Int(imgdata[i*4 + 2])
            a = Int(imgdata[i*4 + 3])
            minDist = 2147483647
            closestMean = 0
            for j in 0..<k {
                w = abs(Int(means[j*4]) - r)
                x = abs(Int(means[j*4]+1) - g)
                y = abs(Int(means[j*4]+2) - b)
                var l = Int(means[j*4]+3)
                z = abs(l - a)
                
                dist = (w^^2) + (x^^2) + (y^^2) + (z^^2)
                if (dist < minDist){
                    closestMean = j
                    minDist = dist
                }
                
            }
            
            assignments[i] = UInt32(closestMean)
        }
        
        
        let end = CACurrentMediaTime()
        print(end - middle)
    }
    
    
    
    func testImageChange() {
        let w = Int((picture.image?.size.width)!)
        let h = Int((picture.image?.size.height)!)
        print("Width: \(w) Height: \(h)")
        let width = max(w,h)
        let height = min(w,h)
        
        let bitsPerComponent:Int = 8
        let bitsPerPixel:Int = 32
        self.img = picture.image?.CGImage!
        self.pixelData = CGDataProviderCopyData(CGImageGetDataProvider(img))
        self.imgdata = CFDataGetBytePtr(pixelData)
        self.lengthOfData = CFDataGetLength(pixelData)

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

infix operator ^^ { }
func ^^ (radix: Int, power: Int) -> Int {
    return Int(pow(Double(radix), Double(power)))
}

