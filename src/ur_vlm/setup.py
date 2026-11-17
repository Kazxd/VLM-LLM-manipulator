from setuptools import setup
package_name = 'ur_vlm'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='you', maintainer_email='you@example.com',
    description='VLM-assisted scene understanding', license='MIT',
    entry_points={'console_scripts': [
        'vlm_node = ur_vlm.vlm_node:main',
        'vlm_bridge = ur_vlm.vlm_bridge:main',
        'vlm_check = ur_vlm.vlm_check:main']},
)
