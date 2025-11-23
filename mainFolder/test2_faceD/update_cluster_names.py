import json
import shutil
from pathlib import Path
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterNameUpdater:
    def __init__(self, embeddings_dir: str = "embeddings"):
        """Initialize cluster name updater."""
        self.embeddings_dir = Path(embeddings_dir)
        self.clusters = {}
        self._load_clusters()
        
        # Create UI
        self.root = tk.Tk()
        self.root.title("Update Cluster Names")
        self.root.geometry("1000x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create cluster list frame
        self.list_frame = ttk.Frame(self.main_frame)
        self.list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create cluster list
        self.cluster_frame = ttk.LabelFrame(self.list_frame, text="Clusters")
        self.cluster_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.cluster_list = tk.Listbox(self.cluster_frame, width=30, selectmode=tk.EXTENDED)
        self.cluster_list.pack(fill=tk.BOTH, expand=True)
        self.cluster_list.bind('<<ListboxSelect>>', self.on_select_cluster)
        
        # Create merge button
        self.merge_button = ttk.Button(self.list_frame, text="Merge Selected Clusters",
                                     command=self.merge_clusters)
        self.merge_button.pack(pady=5)
        
        # Create details frame
        self.details_frame = ttk.LabelFrame(self.main_frame, text="Details")
        self.details_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create image label
        self.image_label = ttk.Label(self.details_frame)
        self.image_label.pack(pady=10)
        
        # Create name entry
        name_frame = ttk.Frame(self.details_frame)
        name_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(name_frame, text="Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Create update button
        self.update_button = ttk.Button(self.details_frame, text="Update Name", 
                                      command=self.update_name)
        self.update_button.pack(pady=5)
        
        # Create preview frame for merge
        self.preview_frame = ttk.LabelFrame(self.details_frame, text="Merge Preview")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Populate cluster list
        self._populate_cluster_list()
    
    def _load_clusters(self):
        """Load cluster information."""
        try:
            for cluster_dir in self.embeddings_dir.glob("cluster_*"):
                if not cluster_dir.is_dir():
                    continue
                
                # Load metadata
                metadata_file = cluster_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Load representative image if exists
                    rep_image_file = cluster_dir / "representative.jpg"
                    rep_image = None
                    if rep_image_file.exists():
                        rep_image = cv2.imread(str(rep_image_file))
                    
                    # Load embeddings
                    embeddings_file = cluster_dir / "raw_embeddings.npy"
                    embeddings = None
                    if embeddings_file.exists():
                        embeddings = np.load(embeddings_file)
                    
                    self.clusters[str(metadata['cluster_id'])] = {
                        'name': metadata['name'],
                        'dir': cluster_dir,
                        'metadata': metadata,
                        'image': rep_image,
                        'embeddings': embeddings
                    }
        
        except Exception as e:
            logger.error(f"Error loading clusters: {str(e)}")
    
    def _populate_cluster_list(self):
        """Populate the cluster list."""
        self.cluster_list.delete(0, tk.END)
        for cluster_id, info in self.clusters.items():
            self.cluster_list.insert(tk.END, f"{cluster_id}: {info['name']}")
    
    def on_select_cluster(self, event):
        """Handle cluster selection."""
        selection = self.cluster_list.curselection()
        if not selection:
            return
        
        # Clear preview frame
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        # Show selected clusters
        row = 0
        for idx in selection:
            cluster_id = self.cluster_list.get(idx).split(':')[0].strip()
            cluster_info = self.clusters.get(cluster_id)
            if not cluster_info or cluster_info['image'] is None:
                continue
            
            # Create frame for this cluster
            cluster_frame = ttk.Frame(self.preview_frame)
            cluster_frame.grid(row=row, column=0, pady=5)
            
            # Show image
            height = 100
            img = cluster_info['image']
            ratio = height / img.shape[0]
            width = int(img.shape[1] * ratio)
            img_resized = cv2.resize(img, (width, height))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            img_label = ttk.Label(cluster_frame)
            img_label.img_tk = img_tk
            img_label.configure(image=img_tk)
            img_label.pack(side=tk.LEFT, padx=5)
            
            # Show info
            info_label = ttk.Label(cluster_frame, 
                                 text=f"Cluster {cluster_id}: {cluster_info['name']}\n"
                                      f"Embeddings: {len(cluster_info['embeddings'])}")
            info_label.pack(side=tk.LEFT, padx=5)
            
            row += 1
        
        # If single selection, update name entry
        if len(selection) == 1:
            cluster_id = self.cluster_list.get(selection[0]).split(':')[0].strip()
            cluster_info = self.clusters.get(cluster_id)
            if cluster_info:
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(0, cluster_info['name'])
    
    def merge_clusters(self):
        """Merge selected clusters."""
        selection = self.cluster_list.curselection()
        if len(selection) < 2:
            messagebox.showwarning("Warning", "Select at least 2 clusters to merge")
            return
        
        # Get selected cluster IDs
        cluster_ids = [self.cluster_list.get(idx).split(':')[0].strip() 
                      for idx in selection]
        
        # Ask for confirmation
        if not messagebox.askyesno("Confirm Merge", 
                                 f"Are you sure you want to merge clusters {', '.join(cluster_ids)}?"):
            return
        
        try:
            # Use first cluster as target
            target_id = cluster_ids[0]
            target_info = self.clusters[target_id]
            
            # Combine embeddings and images
            all_embeddings = [target_info['embeddings']]
            all_images = []
            if target_info['image'] is not None:
                all_images.append(target_info['image'])
            
            # Collect data from other clusters
            for cluster_id in cluster_ids[1:]:
                source_info = self.clusters[cluster_id]
                if source_info['embeddings'] is not None:
                    all_embeddings.append(source_info['embeddings'])
                if source_info['image'] is not None:
                    all_images.append(source_info['image'])
            
            # Combine embeddings
            combined_embeddings = np.concatenate(all_embeddings)
            
            # Update target cluster
            np.save(target_info['dir'] / "raw_embeddings.npy", combined_embeddings)
            
            # Update metadata
            target_info['metadata']['num_embeddings'] = len(combined_embeddings)
            with open(target_info['dir'] / "metadata.json", 'w') as f:
                json.dump(target_info['metadata'], f, indent=2)
            
            # Save best quality image as representative
            if all_images:
                best_image = max(all_images, key=lambda img: np.var(img))
                cv2.imwrite(str(target_info['dir'] / "representative.jpg"), best_image)
            
            # Remove other clusters
            for cluster_id in cluster_ids[1:]:
                source_info = self.clusters[cluster_id]
                shutil.rmtree(source_info['dir'])
                del self.clusters[cluster_id]
            
            # Reload clusters and update UI
            self._load_clusters()
            self._populate_cluster_list()
            
            messagebox.showinfo("Success", "Clusters merged successfully")
            
        except Exception as e:
            logger.error(f"Error merging clusters: {str(e)}")
            messagebox.showerror("Error", f"Failed to merge clusters: {str(e)}")
    
    def update_name(self):
        """Update cluster name."""
        selection = self.cluster_list.curselection()
        if not selection or len(selection) != 1:
            messagebox.showwarning("Warning", "Select exactly one cluster to rename")
            return
        
        # Get cluster ID and new name
        cluster_id = self.cluster_list.get(selection[0]).split(':')[0].strip()
        new_name = self.name_entry.get().strip()
        
        if not new_name:
            return
        
        try:
            cluster_info = self.clusters[cluster_id]
            
            # Update metadata
            metadata = cluster_info['metadata']
            metadata['name'] = new_name
            
            # Save metadata
            metadata_file = cluster_info['dir'] / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update clusters dict
            cluster_info['name'] = new_name
            
            # Update list
            self._populate_cluster_list()
            
            logger.info(f"Updated cluster {cluster_id} name to {new_name}")
            
        except Exception as e:
            logger.error(f"Error updating cluster name: {str(e)}")
            messagebox.showerror("Error", f"Failed to update name: {str(e)}")
    
    def run(self):
        """Run the UI."""
        self.root.mainloop()

def main():
    updater = ClusterNameUpdater()
    updater.run()

if __name__ == "__main__":
    main() 