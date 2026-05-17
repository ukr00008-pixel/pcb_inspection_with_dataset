import requests
import os

def test_api():
    # Test root endpoint
    print("Testing root endpoint...")
    try:
        response = requests.get('http://localhost:8001/')
        print(f"GET /: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

    # Test with good image
    print("\nTesting with good PCB image...")
    good_image_path = r'd:\pcb_inspection_with_dataset\pcb_inspection\data\pcb\good\good_0000.jpg'
    if os.path.exists(good_image_path):
        with open(good_image_path, 'rb') as f:
            files = {'file': ('good_0000.jpg', f, 'image/jpeg')}
            response = requests.post('http://localhost:8001/inspect', files=files)
            print(f"POST /inspect (good): {response.status_code}")
            print(response.json())
    else:
        print("Good image not found")

    # Test with defective image
    print("\nTesting with defective PCB image...")
    defect_image_path = r'd:\pcb_inspection_with_dataset\pcb_inspection\data\pcb_labeled\images\test\cold_joint_0090.jpg'
    if os.path.exists(defect_image_path):
        with open(defect_image_path, 'rb') as f:
            files = {'file': ('cold_joint_0090.jpg', f, 'image/jpeg')}
            response = requests.post('http://localhost:8001/inspect', files=files)
            print(f"POST /inspect (defect): {response.status_code}")
            print(response.json())
    else:
        print("Defect image not found")

if __name__ == "__main__":
    test_api()