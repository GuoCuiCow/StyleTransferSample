/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.styletransfer

import android.Manifest
import android.app.Activity
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.Bundle
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.animation.AnimationUtils
import android.view.animation.BounceInterpolator
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import androidx.lifecycle.ViewModelProviders
import com.bumptech.glide.Glide
import com.bumptech.glide.load.engine.bitmap_recycle.BitmapPool
import com.bumptech.glide.load.resource.bitmap.BitmapTransformation
import com.bumptech.glide.request.RequestOptions
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.android.synthetic.main.bottom_sheet_layout.*
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.async
import org.tensorflow.lite.examples.styletransfer.camera.CameraFragment
import java.io.File
import java.nio.charset.Charset
import java.security.MessageDigest
import java.util.concurrent.Executors

// This is an arbitrary number we are using to keep tab of the permission
// request. Where an app has multiple context for requesting permission,
// this can help differentiate the different contexts
private const val REQUEST_CODE_PERMISSIONS = 10

// This is an array of all the permission specified in the manifest
private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA,Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE)

private const val TAG = "MainActivity"

class MainActivity :
    AppCompatActivity(),
    StyleFragment.OnListFragmentInteractionListener,
    CameraFragment.OnCaptureFinished {

    private var isRunningModel = false
    private val stylesFragment: StyleFragment = StyleFragment()
    private var selectedStyle: String = ""

    private lateinit var cameraFragment: CameraFragment
    private lateinit var viewModel: MLExecutionViewModel
    private lateinit var rerunButton: Button

    private var lastSavedFile = ""
    private var useGPU = false
    private lateinit var styleTransferModelExecutor: StyleTransferModelExecutor
    private val inferenceThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val mainScope = MainScope()

    private var lensFacing = CameraCharacteristics.LENS_FACING_FRONT


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayShowTitleEnabled(false)

        // Request camera permissions
        if (allPermissionsGranted()) {
            addCameraFragment()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }
        pick_image_button.apply {
            setOnClickListener {
                getImage()
            }
        }
        viewModel = ViewModelProviders.of(this)
            .get(MLExecutionViewModel::class.java)

        viewModel.styledBitmap.observe(
            this,
            Observer { resultImage ->
                if (resultImage != null) {
                    updateUIWithResults(resultImage)
                }
            }
        )

        mainScope.async(inferenceThread) {
            styleTransferModelExecutor = StyleTransferModelExecutor(this@MainActivity, useGPU)
            Log.d(TAG, "Executor created")
        }

        switch_use_gpu.setOnCheckedChangeListener { _, isChecked ->
            useGPU = isChecked
            // Disable control buttons to avoid running model before initialization
            enableControls(false)

            // Reinitialize TF Lite models with new GPU setting
            mainScope.async(inferenceThread) {
                styleTransferModelExecutor.close()
                styleTransferModelExecutor = StyleTransferModelExecutor(this@MainActivity, useGPU)

                // Re-enable control buttons
                runOnUiThread { enableControls(true) }
            }
        }

        rerunButton = findViewById(R.id.rerun_button)
        rerunButton.setOnClickListener {
            startRunningModel()
        }

        style_imageview.setOnClickListener {
            if (!isRunningModel) {
                stylesFragment.show(supportFragmentManager, "StylesFragment")
            }
        }

        progress_circular.visibility = View.INVISIBLE
        lastSavedFile = getLastTakenPicture()
        setImageView(original_imageview, lastSavedFile)

        animateCameraButton()
        setupControls()
        enableControls(true)

        Log.d(TAG, "finished onCreate!!")
    }

    private fun getImage() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
        }
        startActivityForResult(intent, 11101)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 11101) {
              data?.data?.apply {
                lastSavedFile = FileUtils.getFilePathByUri(baseContext,this)!!
                Log.e(TAG,lastSavedFile)
                setImageView(original_imageview, lastSavedFile)
              }
            }
        }
    }

    private fun animateCameraButton() {
        val animation = AnimationUtils.loadAnimation(this, R.anim.scale_anim)
        animation.interpolator = BounceInterpolator()
        capture_button.animation = animation
        capture_button.animation.start()
    }

    private fun setImageView(imageView: ImageView, image: Bitmap) {
        Glide.with(baseContext)
            .load(image)
            .override(512, 512)
            .fitCenter()
            .into(imageView)
    }

    private fun setImageView(imageView: ImageView, imagePath: String) {
        Glide.with(baseContext)
            .asBitmap()
            .load(imagePath)
            .override(512, 512)
            .apply(RequestOptions().transform(CropTop()))
            .into(imageView)
    }

    private fun updateUIWithResults(modelExecutionResult: ModelExecutionResult) {
        progress_circular.visibility = View.INVISIBLE
        result_imageview.visibility = View.VISIBLE
        setImageView(result_imageview, modelExecutionResult.styledImage)
        val logText: TextView = findViewById(R.id.log_view)
        logText.text = modelExecutionResult.executionLog
        enableControls(true)
        horizontal_scroll_view.fullScroll(ScrollView.FOCUS_RIGHT)
    }

    private fun enableControls(enable: Boolean) {
        isRunningModel = !enable
        rerunButton.isEnabled = enable
        capture_button.isEnabled = enable
    }

    private fun setupControls() {
        capture_button.setOnClickListener {
            it.clearAnimation()
            cameraFragment.takePicture()
        }

        findViewById<ImageButton>(R.id.toggle_button).setOnClickListener {
            lensFacing = if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
                CameraCharacteristics.LENS_FACING_FRONT
            } else {
                CameraCharacteristics.LENS_FACING_BACK
            }
            cameraFragment.setFacingCamera(lensFacing)
            addCameraFragment()
        }
    }

    private fun addCameraFragment() {
        cameraFragment = CameraFragment.newInstance()
        cameraFragment.setFacingCamera(lensFacing)
        supportFragmentManager.popBackStack()
        supportFragmentManager.beginTransaction()
            .replace(R.id.view_finder, cameraFragment)
            .commit()
    }

    /**
     * Process result from permission request dialog box, has the request
     * been granted? If yes, start Camera. Otherwise display a toast
     */
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                addCameraFragment()
              view_finder.post { setupControls() }
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    /**
     * Check if all permission specified in the manifest have been granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onCaptureFinished(file: File) {
        val msg = "Photo capture succeeded: ${file.absolutePath}"
        Log.d(TAG, msg)

        lastSavedFile = file.absolutePath
        setImageView(original_imageview, lastSavedFile)
    }

    // And update once new picture is taken?
    // Alternatively we can provide user an ability to select any of taken photos
    private fun getLastTakenPicture(): String {
        val directory = baseContext.filesDir // externalMediaDirs.first()
        var files =
            directory.listFiles()?.filter { file -> file.absolutePath.endsWith(".jpg") }?.sorted()
        if (files == null || files.isEmpty()) {
            Log.d(TAG, "there is no previous saved file")
            return ""
        }

        val file = files.last()
        Log.d(TAG, "lastsavedfile: " + file.absolutePath)
        return file.absolutePath
    }

    override fun onListFragmentInteraction(item: String) {
        Log.d(TAG, item)
        selectedStyle = item
        stylesFragment.dismiss()

        startRunningModel()
    }

    private fun getUriFromAssetThumb(thumb: String): String {
        return "file:///android_asset/thumbnails/$thumb"
    }

    private fun startRunningModel() {
        if (!isRunningModel && lastSavedFile.isNotEmpty() && selectedStyle.isNotEmpty()) {
            val chooseStyleLabel: TextView = findViewById(R.id.choose_style_text_view)
            chooseStyleLabel.visibility = View.GONE
            enableControls(false)
            setImageView(style_imageview, getUriFromAssetThumb(selectedStyle))
            result_imageview.visibility = View.INVISIBLE
            progress_circular.visibility = View.VISIBLE
            viewModel.onApplyStyle(
                baseContext, lastSavedFile, selectedStyle, styleTransferModelExecutor,
                inferenceThread
            )
        } else {
            Toast.makeText(this, "Previous Model still running", Toast.LENGTH_SHORT).show()
        }
    }

    // this transformation is necessary to show the top square of the image as the model
    // will work on this part only, making the preview and the result show the same base
    class CropTop : BitmapTransformation() {
        override fun transform(
            pool: BitmapPool,
            toTransform: Bitmap,
            outWidth: Int,
            outHeight: Int
        ): Bitmap {
            return if (toTransform.width == outWidth && toTransform.height == outHeight) {
                toTransform
            } else ImageUtils.scaleBitmapAndKeepRatio(toTransform, outWidth, outHeight)
        }

        override fun equals(other: Any?): Boolean {
            return other is CropTop
        }

        override fun hashCode(): Int {
            return ID.hashCode()
        }

        override fun updateDiskCacheKey(messageDigest: MessageDigest) {
            messageDigest.update(ID_BYTES)
        }

        companion object {
            private const val ID = "org.tensorflow.lite.examples.styletransfer.CropTop"
            private val ID_BYTES = ID.toByteArray(Charset.forName("UTF-8"))
        }
    }

}
