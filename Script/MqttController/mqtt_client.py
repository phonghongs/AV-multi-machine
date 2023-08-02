import asyncio
from random import randrange
from asyncio_mqtt import Client

async def log_filtered_messages(client, topic_filter):
    async with client.filtered_messages(topic_filter) as messages:
        async for message in messages:
            print(f'[topic_filter="{topic_filter}"]: {message.payload.decode()}')

async def log_unfiltered_messages(client):
    async with client.unfiltered_messages() as messages:
        async for message in messages:
            print(f'[unfiltered]: {message.payload.decode()}')

async def main():
    loop = asyncio.get_event_loop()
    async with Client('192.168.1.51', 1883) as client:
        await client.subscribe('floors/#')

        # You can create any number of message filters
        loop.create_task(log_filtered_messages(client, 'floors/+/humidity'))
        loop.create_task(log_filtered_messages(client, 'floors/rooftop/#'))
        # 👉 Try to add more filters!

        # All messages that doesn't match a filter will get logged here
        loop.create_task(log_unfiltered_messages(client))

        # # Publish a random value to each of these topics
        topics = [
            'floors/basement/humidity',
            'floors/rooftop/humidity',
            'floors/rooftop/illuminance',
            # 👉 Try to add more topics!
        ]
        while True:
            for topic in topics:
                message = randrange(100)
                # print(f'[topic="{topic}"] Publishing message={message}')
                # await client.publish(topic, message, qos=1)
                await asyncio.sleep(2)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
