import numpy as np
import pybullet
import structlog
from arm_1dof.bullet_arm_1dof import BulletArm1Dof
from arm_1dof.robot_arm_1dof import RobotArm1Dof

_log = structlog.get_logger("dataset_gen")


class Robot:
    def __init__(self):
        bullet_world = BulletArm1Dof()
        bullet_world.InitPybullet()

        self.bullet_robot = bullet_world.LoadRobot()
        bullet_world.LoadPlane()

    def _set_EE_pos(self, position_rad) -> np.ndarray:
        """Sets joint to position_rad and returns EE (cartesian) position"""
        pybullet.resetJointState(
            self.bullet_robot._body_id,
            RobotArm1Dof.ELBOW_JOINT_ID,
            position_rad,
        )
        return

    def _capture_state_and_save(self, image_path) -> None:
        from PIL import Image

        _log.debug("setting up camera...")

        camera_target_position = [0.3, 0.3, 1.5]
        camera_position = [0, -1, 1.7]
        up_vector = [0, 0, 1]
        width = 1024
        height = 768
        fov = 60
        aspect = width / height
        near = 0.1
        far = 100
        projection_matrix = pybullet.computeProjectionMatrixFOV(fov, aspect, near, far)
        view_matrix = pybullet.computeViewMatrix(
            camera_position, camera_target_position, up_vector
        )
        _log.debug("getting image...")
        img_arr = pybullet.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        _log.debug("saving image...")
        rgb_buffer = np.array(img_arr[2])
        rgb = rgb_buffer[:, :, :3]  # drop alpha
        Image.fromarray(rgb.astype(np.uint8)).save(image_path)
        _log.info(f"saved input image at {str(image_path)}")


def generate_image(joint, ball, image_path="./test.bmp"):
    robot = Robot()
    robot._set_EE_pos(joint)
    robot._capture_state_and_save(image_path)


if __name__ == "__main__":
    generate_image(1.57079633, None, "./test_start.bmp")
    generate_image(0.34906585, None, "./test_end.bmp")
