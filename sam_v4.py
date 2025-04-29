# this scripts already loaded SAM2 tiny Model
# when we use it only two variable needed: wsi path and use_box;
# in my device, run by(The first is using box in ROI for dectecting, the another is using point, both point and box coordination can adjust): 
## 1)python /Volumes/LANQI/SAM/sam_v4.py --wsi /Volumes/LANQI/SAM/MISC130.tif --direct-load --use-box
## 2)python /Volumes/LANQI/SAM/sam_v4.py --wsi /Volumes/LANQI/SAM/MISC130.tif --direct-load 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import scipy.sparse as sparse
import cv2
import warnings
from pathlib import Path

# Try to import openslide, but don't fail if it's not available
try:
    import openslide
except ImportError:
    warnings.warn("OpenSlide not found. Installing it or using PIL as fallback for small WSIs.")
    openslide = None

# Check if we're running in a notebook (for plotting)
in_notebook = 'ipykernel' in sys.modules

class SAM2WSIWrapper:
    """
    Wrapper for SAM2 to work with Whole Slide Images (WSI) in pathology.
    
    This class handles:
    1. Loading and processing WSIs
    2. Extracting regions of interest (ROIs)
    3. Running SAM2 on these ROIs
    4. Mapping coordinates between ROI and WSI spaces
    5. Saving and visualizing results
    """
    def __init__(self, sam2_checkpoint=None, model_cfg=None, device="cpu", sam2_model=None):
        """
        Initialize SAM2 model for WSI processing.
        
        Args:
            sam2_checkpoint (str, optional): Path to SAM2 checkpoint
            model_cfg (str, optional): Path to model configuration
            device (str): Device to run model on ("cuda" or "cpu")
            sam2_model: Pre-loaded SAM2 model (if provided, checkpoint and config not needed)
        """
        self.device = device
        self.current_roi_info = None
        self.wsi = None
        
        # If a model is directly provided, use it
        if sam2_model is not None:
            print(f"Using provided SAM2 model...")
            self.sam2_model = sam2_model
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            return
        
        # Otherwise, try to load the model from checkpoint and config
        try:
            print(f"Initializing SAM2 model on {device}...")
            
            # Check file existence and permissions first
            if not os.path.exists(model_cfg):
                raise FileNotFoundError(f"Model config file not found: {model_cfg}")
            
            if not os.access(model_cfg, os.R_OK):
                raise PermissionError(f"Cannot read model config file: {model_cfg}")
                
            if not os.path.exists(sam2_checkpoint):
                raise FileNotFoundError(f"Checkpoint file not found: {sam2_checkpoint}")
                
            if not os.access(sam2_checkpoint, os.R_OK):
                raise PermissionError(f"Cannot read checkpoint file: {sam2_checkpoint}")
            
            # Load model
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            print("SAM2 model initialized successfully.")
            
        except (ImportError, FileNotFoundError, PermissionError) as e:
            print(f"Error initializing SAM2 model: {e}")
            print("Make sure paths to model_cfg and sam2_checkpoint are correct and accessible.")
            print("Absolute paths are recommended.")
            raise
        
    def load_wsi(self, wsi_path):
        """
        Load a WSI image.
        
        Args:
            wsi_path (str): Path to the WSI file
            
        Returns:
            The loaded WSI
        """
        print(f"Loading WSI from {wsi_path}...")
        
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
            
        if not os.access(wsi_path, os.R_OK):
            raise PermissionError(f"Cannot read WSI file: {wsi_path}")
            
        try:
            # Try to use OpenSlide if available
            if openslide is not None:
                self.wsi = openslide.OpenSlide(wsi_path)
                print(f"WSI loaded with OpenSlide. Dimensions: {self.wsi.dimensions}")
                self.wsi_backend = "openslide"
            else:
                # Fallback to PIL for smaller images
                self.wsi = Image.open(wsi_path)
                print(f"WSI loaded with PIL. Dimensions: {self.wsi.size}")
                self.wsi_backend = "pil"
                
            return self.wsi
            
        except Exception as e:
            print(f"Error loading WSI: {e}")
            raise
            
    def _get_wsi_dimensions(self):
        """Get WSI dimensions depending on backend."""
        if self.wsi_backend == "openslide":
            return self.wsi.dimensions
        else:  # PIL
            return self.wsi.size
    
    def extract_roi(self, box_coords, level=0):
        """
        Extract a region of interest (ROI) from the WSI.
        
        Args:
            box_coords (tuple or list): Coordinates of the ROI in WSI space (x_min, y_min, x_max, y_max)
            level (int): Pyramid level to extract from (only used with OpenSlide)
            
        Returns:
            numpy.ndarray: The extracted ROI as an RGB image
        """
        x_min, y_min, x_max, y_max = box_coords
        width = x_max - x_min
        height = y_max - y_min
        
        print(f"Extracting ROI at level {level} with coordinates: {box_coords}")
        
        try:
            # Extract the region from the WSI
            if self.wsi_backend == "openslide":
                roi = self.wsi.read_region((x_min, y_min), level, (width, height))
                roi = np.array(roi.convert("RGB"))
            else:  # PIL
                # For PIL, we crop directly (level is ignored)
                roi = self.wsi.crop((x_min, y_min, x_max, y_max))
                roi = np.array(roi.convert("RGB"))
                
            # Store ROI info for later use
            self.current_roi_info = {
                "box": box_coords,
                "level": level,
                "offset": (x_min, y_min),
                "size": (width, height)
            }
            
            return roi
            
        except Exception as e:
            print(f"Error extracting ROI: {e}")
            raise
    
    def process_roi(self, roi):
        """
        Set the ROI image for SAM2 processing.
        
        Args:
            roi (numpy.ndarray): RGB image of the ROI
        """
        print("Setting ROI image for SAM2 processing...")
        self.predictor.set_image(roi)
        print("ROI image set successfully.")
        
    def detect_in_roi(self, point_or_box, mode="point", multimask_output=False, point_labels=None):
        """
        Detect objects in the ROI using either point or box prompts.
        
        Args:
            point_or_box: Either point coordinates [x, y] or box coordinates [x_min, y_min, x_max, y_max]
            mode (str): "point" or "box"
            multimask_output (bool): Whether to output multiple mask options
            point_labels (numpy.ndarray): Labels for points (1 for foreground, 0 for background)
            
        Returns:
            tuple: (masks, scores, logits) from SAM2
        """
        try:
            if mode == "point":
                if not isinstance(point_or_box, np.ndarray):
                    point_or_box = np.array([point_or_box])
                
                if point_labels is None:
                    point_labels = np.ones(len(point_or_box))
                
                print(f"Detecting with {len(point_or_box)} points in ROI...")
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_or_box,
                    point_labels=point_labels,
                    multimask_output=multimask_output
                )
            elif mode == "box":
                if not isinstance(point_or_box, np.ndarray):
                    point_or_box = np.array(point_or_box)
                
                print(f"Detecting with box {point_or_box} in ROI...")
                masks, scores, logits = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=point_or_box.reshape(1, -1),
                    multimask_output=multimask_output
                )
            else:
                raise ValueError("Mode must be either 'point' or 'box'")
            
            return masks, scores, logits
            
        except Exception as e:
            print(f"Error during detection: {e}")
            raise
    
    def roi_to_wsi_coordinates(self, roi_coords):
        """
        Convert coordinates from ROI space to WSI space.
        
        Args:
            roi_coords (tuple or numpy.ndarray): Coordinates in ROI space
            
        Returns:
            tuple or numpy.ndarray: Coordinates in WSI space
        """
        if self.current_roi_info is None:
            raise ValueError("No ROI has been processed yet")
        
        x_offset, y_offset = self.current_roi_info["offset"]
        
        if isinstance(roi_coords, np.ndarray):
            if roi_coords.ndim == 1:  # Single point
                return np.array([roi_coords[0] + x_offset, roi_coords[1] + y_offset])
            elif roi_coords.ndim == 2:  # Multiple points
                result = roi_coords.copy()
                result[:, 0] += x_offset
                result[:, 1] += y_offset
                return result
            elif roi_coords.ndim > 2:  # Masks or other structures
                raise ValueError("Conversion not implemented for this data structure")
        elif isinstance(roi_coords, (list, tuple)) and len(roi_coords) == 4:  # Box coordinates
            x_min, y_min, x_max, y_max = roi_coords
            return (x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset)
        else:
            raise ValueError("Unsupported coordinate format")
    
    def get_mask_contours(self, mask):
        """
        Extract contours from a binary mask.
        
        Args:
            mask (numpy.ndarray): Binary mask
            
        Returns:
            list: List of contour points in (x, y) format
        """
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to list of (x, y) points
        contour_points = []
        for contour in contours:
            points = contour.reshape(-1, 2)
            contour_points.append(points)
            
        return contour_points
    
    def contours_roi_to_wsi(self, contours):
        """
        Convert contour coordinates from ROI space to WSI space.
        
        Args:
            contours (list): List of contours in ROI space
            
        Returns:
            list: List of contours in WSI space
        """
        if self.current_roi_info is None:
            raise ValueError("No ROI has been processed yet")
        
        x_offset, y_offset = self.current_roi_info["offset"]
        
        wsi_contours = []
        for contour in contours:
            wsi_contour = contour.copy()
            wsi_contour[:, 0] += x_offset
            wsi_contour[:, 1] += y_offset
            wsi_contours.append(wsi_contour)
            
        return wsi_contours
    
    def save_masks_sparse(self, masks, filename):
        """
        Save masks as sparse matrices.
        
        Args:
            masks (numpy.ndarray): Binary masks to save
            filename (str): Path to save the sparse matrices
        """
        try:
            sparse_masks = []
            
            for i, mask in enumerate(masks):
                # Convert to sparse matrix
                sparse_mask = sparse.csr_matrix(mask)
                sparse_masks.append(sparse_mask)
            
            # Save using pickle
            with open(filename, 'wb') as f:
                pickle.dump(sparse_masks, f)
            
            print(f"Saved {len(sparse_masks)} masks to {filename}")
            
        except Exception as e:
            print(f"Error saving masks: {e}")
            raise
        
    def save_results(self, masks, scores, wsi_contours, filename, wsi_box=None):
        """
        Save detection results including masks, scores, and WSI contours.
        
        Args:
            masks (numpy.ndarray): Binary masks
            scores (numpy.ndarray): Confidence scores
            wsi_contours (list): Contours in WSI coordinates
            filename (str): Path to save the results
        """
        try:
            # Convert masks to sparse matrices
            sparse_masks = [sparse.csr_matrix(mask) for mask in masks]
            
            results = {
                "sparse_masks": sparse_masks,
                "scores": scores,
                "wsi_contours": wsi_contours,
                "roi_info": self.current_roi_info,
                "wsi_box": wsi_box
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Saved results to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            raise
    
    def visualize_roi_detection(self, roi, masks, scores, point_coords=None, box_coords=None, 
                              input_labels=None, borders=True, figsize=(12, 12), save_path=None):
        """
        Visualize detection results within the ROI.
        
        Args:
            roi (numpy.ndarray): RGB image of the ROI
            masks (numpy.ndarray): Binary masks
            scores (numpy.ndarray): Confidence scores
            point_coords (numpy.ndarray): Point prompts
            box_coords (numpy.ndarray): Box prompts
            input_labels (numpy.ndarray): Point labels
            borders (bool): Whether to draw borders around masks
            figsize (tuple): Figure size
            save_path (str, optional): Path to save visualization
        """
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=figsize)
            plt.imshow(roi)
            self._show_mask(mask, plt.gca(), borders=borders)
            
            if point_coords is not None and input_labels is not None:
                self._show_points(point_coords, input_labels, plt.gca())
                
            if box_coords is not None:
                self._show_box(box_coords, plt.gca())
                
            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            else:
                plt.title(f"Detection Score: {score:.3f}", fontsize=18)
                
            plt.axis('off')
            
            if box_coords is not None:
                self._show_box(box_coords, plt.gca())
            
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                mask_save_path = f"{os.path.splitext(save_path)[0]}_mask{i}{os.path.splitext(save_path)[1]}"
                plt.savefig(mask_save_path, bbox_inches='tight')
                print(f"Saved visualization to {mask_save_path}")
            
            plt.show() if in_notebook else plt.close()
    
    def visualize_wsi_detection(self, masks, wsi_contours, figsize=(12, 12), save_path=None):
        """
        Visualize detection results on the WSI (simplified representation).
        
        Args:
            masks (numpy.ndarray): Binary masks
            wsi_contours (list): Contours in WSI coordinates
            figsize (tuple): Figure size
            save_path (str, optional): Path to save visualization
        """
        # Get ROI boundaries in WSI space
        x_min, y_min, x_max, y_max = self.current_roi_info["box"]
        
        plt.figure(figsize=figsize)
        
        # Draw ROI boundary
        plt.plot([x_min, x_max, x_max, x_min, x_min], 
                 [y_min, y_min, y_max, y_max, y_min], 'b-', linewidth=2)
        
        # Draw contours
        for contour in wsi_contours:
            plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
        
        plt.title("Detection Results in WSI Coordinates", fontsize=18)
        plt.gca().set_aspect('equal')

        
        plt.gca().invert_yaxis()  #4.2 debug trying
        plt.gca().set_aspect('equal', adjustable='box')  

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved WSI visualization to {save_path}")
        
        plt.show() if in_notebook else plt.close()
    
    def _show_mask(self, mask, ax, random_color=False, borders=True):
        """Helper method to display a mask."""
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
            
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
            
        ax.imshow(mask_image)
    
    def _show_points(self, coords, labels, ax, marker_size=375):
        """Helper method to display points."""
        try:
            pos_points = coords[labels==1]
            neg_points = coords[labels==0]
            
            if len(pos_points) > 0:
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                          s=marker_size, edgecolor='white', linewidth=1.25)
            
            if len(neg_points) > 0:
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                          s=marker_size, edgecolor='white', linewidth=1.25)
        except Exception as e:
            print(f"Error showing points: {e}")
    
    def _show_box(self, box, ax):
        """Helper method to display a box."""
        try:
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        except Exception as e:
            print(f"Error showing box: {e}")

# Example usage function with better error handling
def run_sam2_wsi_example(wsi_path, sam2_checkpoint=None, model_cfg=None, device="cpu", 
                        sam2_model=None, output_dir="./output", use_box=False, roi_point = None,roi_labels = None,wsi_box = None):
    """
    Example workflow for using SAM2 with WSI images.
    
    Args:
        wsi_path (str): Path to WSI file
        sam2_checkpoint (str, optional): Path to SAM2 checkpoint
        model_cfg (str, optional): Path to model configuration
        device (str): Device to run on
        sam2_model: Pre-loaded SAM2 model (optional)
        output_dir (str): Directory to save outputs
        use_box (bool): use box or point in ROI
    """
    
    if output_dir is None:
        output_dir = "./output"
        
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize the SAM2 WSI wrapper
        print("Initializing SAM2 WSI wrapper...")
        sam2_wsi = SAM2WSIWrapper(
            sam2_checkpoint=sam2_checkpoint, 
            model_cfg=model_cfg, 
            device=device,
            sam2_model=sam2_model
        )
        
        # Load the WSI
        wsi = sam2_wsi.load_wsi(wsi_path)
        
        # Define a ROI in WSI coordinates
        # adjusted based on specific WSI, 
        roi_box = (8000, 7500, 12000, 10000)  # (x_min, y_min, x_max, y_max) in case for MISC130.tif, the size is 17222 × 17629
        #roi_box = (2000, 2500, 2500, 3500) #3907 × 5570 for SU146.tif
        # Extract the ROI
        print("Extracting ROI...")
        roi = sam2_wsi.extract_roi(roi_box, level=0)
        
        # Process the ROI with SAM2
        print("Processing ROI with SAM2...")
        sam2_wsi.process_roi(roi)
        
        if use_box:
            
            roi_box_prompt = np.array([500, 500, 1500, 1500])  # (x_min, y_min, x_max, y_max)
            
            print("Running detection with box prompt...")
            masks, scores, logits = sam2_wsi.detect_in_roi(
                roi_box_prompt, 
                mode="box", 
                multimask_output=True
            )
            
            wsi_box = sam2_wsi.roi_to_wsi_coordinates(roi_box_prompt)
        else:
            roi_point = np.array([[500, 500]]) 
            roi_labels = np.array([1])
            
            print("Running detection with point prompt...")
            masks, scores, logits = sam2_wsi.detect_in_roi(
                roi_point, 
                mode="point", 
                multimask_output=True,
                point_labels=roi_labels
            )
            wsi_box = None
        
        # Visualize the detection results in ROI space
        print("Visualizing ROI detection results...")
        vis_path = os.path.join(str(output_dir), "roi_detection.png")
        
            # 修改为：
        if use_box:
            # 使用box提示时不需要point参数
            sam2_wsi.visualize_roi_detection(
                roi, 
                masks, 
                scores, 
                box_coords=roi_box_prompt,
                save_path=vis_path
                )
        else:
        # 使用point提示时传递point参数
            sam2_wsi.visualize_roi_detection(
                roi, 
                masks, 
                scores, 
                point_coords=roi_point,
                input_labels=roi_labels,
                save_path=vis_path
            )

        # Get the best mask
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Extract contours from the mask
        print("Extracting mask contours...")
        roi_contours = sam2_wsi.get_mask_contours(best_mask)
        
        # Convert contours to WSI coordinates
        print("Converting to WSI coordinates...")
        wsi_contours = sam2_wsi.contours_roi_to_wsi(roi_contours)
        
        # Visualize detection results in WSI space
        print("Visualizing WSI detection results...")

        wsi_vis_path = os.path.join(str(output_dir), "wsi_detection.png")
        sam2_wsi.visualize_wsi_detection(
            [best_mask], 
            wsi_contours,
            save_path=wsi_vis_path
        )
        
        # Save results
        print("Saving results...")
        
        results_path = os.path.join(str(output_dir), "sam2_wsi_detection_results.pkl")
        sam2_wsi.save_results(
            masks=[best_mask], 
            scores=np.array([scores[best_mask_idx]]), 
            wsi_contours=wsi_contours, 
            wsi_box=wsi_box, # FOR BOX WITHIN ROI
            filename=results_path
        )
        
        print("SAM2 WSI processing completed successfully!")
        return sam2_wsi, masks, scores, wsi_contours
        
    except Exception as e:
        print(f"Error during SAM2 WSI processing: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Check file paths and permissions")
        print("2. Verify SAM2 installation and dependencies")
        print("3. Make sure the WSI file is valid and accessible")
        print("4. Consider using a different pyramid level for large WSIs")
        raise


# Main function
if __name__ == "__main__":
    import argparse
    import pdb  # 添加调试模块
    
    parser = argparse.ArgumentParser(description="SAM2 WSI Processing")
    parser.add_argument("--wsi", type=str, required=True, help="Path to WSI file")
    parser.add_argument("--checkpoint", type=str, help="Path to SAM2 checkpoint")
    parser.add_argument("--config", type=str, help="Path to SAM2 model config")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cuda/cpu)")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--direct-load", action="store_true", help="Load model directly without config")
    parser.add_argument("--use-box", action="store_true", help="Use box prompt instead of point")

    
    args = parser.parse_args()
    
    try:
        # Check for required files
        if not os.path.exists(args.wsi):
            raise FileNotFoundError(f"WSI file not found: {args.wsi}")
    
        if args.direct_load:
            print("Loading SAM2 model directly...")
            
            try:
                import hydra
                from hydra.core.global_hydra import GlobalHydra
                
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                
                config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam2_configs')
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                    
                    with open(os.path.join(config_dir, '__init__.py'), 'w') as f:
                        pass
                
                hydra.initialize_config_dir(config_dir, version_base='1.2')
                
                from sam2.build_sam import build_sam2
                import torch
                
                # load model
                sam2_model = build_sam2('sam2_hiera_t.yaml', '/Volumes/LANQI/SAM/sam2_hiera_tiny.pt', device=args.device)
                
                # if the load fails，try strict=False
                try:
                    state_dict = torch.load('/Volumes/LANQI/SAM/sam2_hiera_tiny.pt')
                    sam2_model.load_state_dict(state_dict, strict=False)
                except Exception as load_err:
                    print(f"Warning when loading state dict: {load_err}")
                
                # 确保输出目录存在
                os.makedirs(args.output, exist_ok=True)
                
                # 运行example
                run_sam2_wsi_example(
                    wsi_path=args.wsi,
                    sam2_model=sam2_model,
                    device=args.device,
                    output_dir=args.output,
                    use_box=args.use_box
                )
            except Exception as inner_e:
                print(f"Error in direct-load mode: {inner_e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            # Load using config and checkpoint
            if not args.checkpoint or not args.config:
                print("Error: Both checkpoint and config must be provided unless using --direct-load")
                sys.exit(1)
                
            # Convert to absolute paths to avoid permission issues
            checkpoint_path = os.path.abspath(args.checkpoint)
            config_path = os.path.abspath(args.config)
                
            run_sam2_wsi_example(
                wsi_path=args.wsi,
                sam2_checkpoint=checkpoint_path,
                model_cfg=config_path,
                device=args.device,
                output_dir=args.output,
                use_box=args.use_box
            )
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
