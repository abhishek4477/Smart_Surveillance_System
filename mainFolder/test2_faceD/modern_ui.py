import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path
import time

# Set appearance mode and default color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ModernUI:
    def __init__(self):
        # Color scheme
        self.COLORS = {
            'primary': '#1f538d',
            'secondary': '#14375e',
            'accent': '#00a8e8',
            'success': '#2ecc71',
            'warning': '#e67e22',
            'error': '#e74c3c',
            'text_primary': '#ffffff',
            'text_secondary': '#b2bec3'
        }
        
        # Load and cache images
        self.images = {}
        self._load_images()
    
    def _load_images(self):
        """Load and prepare images for UI elements"""
        image_dir = Path(__file__).parent / "assets"
        if not image_dir.exists():
            image_dir.mkdir()
            
        # Define default icons if needed
        default_icons = {
            'attendance': '🎓',
            'crowd': '👥',
            'behavior': '👀',
            'settings': '⚙️'
        }
        
        # Try to load images, use defaults if not found
        for name, default in default_icons.items():
            img_path = image_dir / f"{name}.png"
            if img_path.exists():
                self.images[name] = ctk.CTkImage(
                    light_image=Image.open(img_path),
                    dark_image=Image.open(img_path),
                    size=(32, 32)
                )
            else:
                # Create text-based fallback
                self.images[name] = default

    def create_main_window(self, root):
        """Create and style the main window"""
        # Configure main window
        root.title("OmniSight")
        root.geometry("1200x800")
        
        # Create main container
        container = ctk.CTkFrame(root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self._create_header(container)
        
        # Create system cards container
        cards_frame = ctk.CTkFrame(container, fg_color="transparent")
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configure grid for responsive layout
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="column")
        cards_frame.grid_rowconfigure(0, weight=1)
        
        # Create system cards
        self._create_system_card(cards_frame, 0, "Attendance System", 
                               "Track attendance using facial recognition",
                               self.images['attendance'])
        
        self._create_system_card(cards_frame, 1, "Crowd Management", 
                               "Monitor and analyze crowd density",
                               self.images['crowd'])
        
        self._create_system_card(cards_frame, 2, "Behavior Analysis", 
                               "Analyze and log behavioral patterns",
                               self.images['behavior'])
        
        # Create status bar
        self._create_status_bar(container)
        
        return container

    def _create_header(self, parent):
        """Create modern header with title and subtitle"""
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill=tk.X, pady=(0, 20))
        
        title = ctk.CTkLabel(
            header,
            text="OmniSight",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=self.COLORS['accent']
        )
        title.pack(pady=(0, 5))
        
        subtitle = ctk.CTkLabel(
            header,
            text="Smart Surveillance System",
            font=ctk.CTkFont(size=16),
            text_color=self.COLORS['text_secondary']
        )
        subtitle.pack()

    def _create_system_card(self, parent, column, title, description, icon):
        """Create a modern card for each system"""
        card = ctk.CTkFrame(parent, corner_radius=15)
        card.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")
        
        # Add hover effect
        def on_enter(e):
            card.configure(fg_color=self.COLORS['secondary'])
        def on_leave(e):
            card.configure(fg_color=("gray90", "gray10"))
            
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        
        # Icon/Image
        if isinstance(icon, str):
            # Use text-based icon
            icon_label = ctk.CTkLabel(
                card,
                text=icon,
                font=ctk.CTkFont(size=48),
                text_color=self.COLORS['accent']
            )
        else:
            # Use image icon
            icon_label = ctk.CTkLabel(
                card,
                image=icon,
                text=""
            )
        icon_label.pack(pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=self.COLORS['text_primary']
        )
        title_label.pack(pady=10)
        
        # Description
        desc_label = ctk.CTkLabel(
            card,
            text=description,
            font=ctk.CTkFont(size=14),
            text_color=self.COLORS['text_secondary'],
            wraplength=200
        )
        desc_label.pack(pady=10)
        
        # Launch button
        button = ctk.CTkButton(
            card,
            text="Launch",
            font=ctk.CTkFont(size=14),
            corner_radius=25,
            height=35
        )
        button.pack(pady=20)

    def _create_status_bar(self, parent):
        """Create modern status bar"""
        status_frame = ctk.CTkFrame(parent, height=30, fg_color=self.COLORS['secondary'])
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status text
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=self.COLORS['text_secondary']
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Time display
        self.time_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=self.COLORS['text_secondary']
        )
        self.time_label.pack(side=tk.RIGHT, padx=10)
        
        # Update time
        self._update_time()

    def _update_time(self):
        """Update time display in status bar"""
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.configure(text=current_time)
        self.time_label.after(1000, self._update_time)

    def update_status(self, message):
        """Update status bar message"""
        self.status_label.configure(text=message)

    def create_attendance_window(self, root):
        """Create modern attendance system window"""
        container = ctk.CTkFrame(root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add attendance-specific UI elements here
        return container

    def create_crowd_window(self, root):
        """Create modern crowd management window"""
        container = ctk.CTkFrame(root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add crowd management-specific UI elements here
        return container

    def create_behavior_window(self, root):
        """Create modern behavior analysis window"""
        container = ctk.CTkFrame(root, fg_color="transparent")
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add behavior analysis-specific UI elements here
        return container

# Example usage
if __name__ == "__main__":
    # Create test window
    root = ctk.CTk()
    ui = ModernUI()
    main_container = ui.create_main_window(root)
    root.mainloop() 